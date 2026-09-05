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

import base64
import html as html_module
import io
import os
import sys
import tempfile
import time
import traceback as tb_module
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    from backend_core.AgentMPC.agents.diagnostics_agent import (
        ERROR_CATEGORY_TITLES, chat_about_issues, generate_diagnostics_report, scan_for_issues,
    )
    DIAGNOSTICS_FEATURE_AVAILABLE = True
    DIAGNOSTICS_FEATURE_ERROR = None
except ImportError as e:
    DIAGNOSTICS_FEATURE_AVAILABLE = False
    DIAGNOSTICS_FEATURE_ERROR = str(e)
try:
    from backend_core.AgentMPC.agents.llm_base import TokenUsageTracker
    TOKEN_TRACKING_AVAILABLE = True
    TOKEN_TRACKING_ERROR = None
except ImportError as e:
    TOKEN_TRACKING_AVAILABLE = False
    TOKEN_TRACKING_ERROR = str(e)
from backend_core.AgentMPC.agents.scenario_presets import (
    SCENARIO_LEVEL_NAMES, apply_scenario_level, nudge_if_starts_at_target, suggested_noise_std,
)
from backend_core.AgentMPC.agents.seed_params import parse_seed_params
from backend_core.AgentMPC.dynamics.base import SystemSimulator
from backend_core.AgentMPC.dynamics.loader import DynamicLoader, DynamicsPluginError
from langgraph.errors import GraphRecursionError
from backend_core.AgentMPC.graph.workflow import build_ui_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentMPC.utils.logging_utils import configure_logging, get_logger

# ============================================================================
# PAGE CONFIG + THEME
# ============================================================================

st.set_page_config(page_title="LabCD · MPC Studio", page_icon=":gear:", layout="wide", initial_sidebar_state="collapsed")
configure_logging()
log = get_logger(__name__)

# Single source of truth for the dashboard's look, so the live page and the
# UI Snapshot export (build_ui_snapshot_html, near render_report_section)
# can never drift apart -- the export literally reuses this same string
# rather than a hand-copied approximation of it.
DASHBOARD_CSS = """<style>
    /* ========================================================================
       LabCD · MPC Studio — app stylesheet

       TYPEFACE (deliberate, do not "fix")
       ------------------------------------------------------------------
       The whole product is set in Times New Roman: the UI here and the
       generated PDF report (see agents/report_pdf.py, which uses reportlab's
       built-in Times faces for the same reason). Every rule below is designed
       around that constraint rather than against it.

       Times has a noticeably small x-height and light stems, so at the sizes
       a UI normally uses it reads thin and cramped. The compensations are:
         * a larger base size (16.5px) and a generous 1.62 line-height;
         * small labels set as letter-spaced uppercase, which is where a serif
           is at its strongest rather than its weakest;
         * every number, parameter and log line in monospace, so the serif
           never has to carry tabular data it is bad at aligning.

       DESIGN TOKENS
       ------------------------------------------------------------------
       The values below used to be written inline, which had produced five
       near-identical panel backgrounds (rgba(20,30,50,.32/.28/.25/.18) and
       rgba(6,10,18,.55)), six border alphas, and two unrelated accent
       palettes -- the blue/indigo family plus leftover Dracula colours
       (#f1fa8c, #50fa7b, #ff5555, #8be9fd). Surfaces and accents are single
       sources of truth now, so panels actually match each other.
       ======================================================================== */
    :root {
        /* surfaces, lightest-on-top */
        --bg:        #0a0d13;
        --surface-1: rgba(255,255,255,0.022);   /* cards, panels           */
        --surface-2: rgba(255,255,255,0.040);   /* raised: metrics, inputs */
        --surface-3: rgba(255,255,255,0.060);   /* hover                   */
        --sunken:    rgba(4,7,13,0.55);         /* logs, code, scroll wells */

        /* borders */
        --line:      rgba(255,255,255,0.075);
        --line-soft: rgba(255,255,255,0.045);
        --line-firm: rgba(255,255,255,0.130);

        /* text */
        --text:      #eaeff8;
        --text-2:    #a3aec2;
        --text-3:    #7d8598;
        --text-4:    #5f6a80;

        /* one accent family + harmonised semantics (no Dracula strays) */
        --accent:    #4d9fff;
        --accent-2:  #818cf8;
        --accent-dim:rgba(77,159,255,0.14);
        --ok:        #3ddc97;
        --warn:      #e8b13d;
        --bad:       #f2617a;
        --info:      #56c8e8;

        /* type scale — chosen for Times' small x-height */
        --t-micro: 0.70rem;   /* letter-spaced uppercase labels */
        --t-small: 0.82rem;
        --t-body:  0.95rem;
        --t-lead:  1.05rem;
        --t-h2:    1.20rem;
        --t-h1:    1.45rem;

        --mono: 'Consolas','SF Mono',Menlo,'Courier New',monospace;
        --radius:    12px;
        --radius-sm: 9px;

        /* one elevation model instead of a different shadow per component */
        --lift:  0 1px 2px rgba(0,0,0,0.28);
        --lift-2:0 6px 20px rgba(0,0,0,0.30);
    }

    html, body, .stApp {
        background: radial-gradient(ellipse 1400px 900px at 15% -10%, rgba(77,159,255,0.035), transparent 60%),
                    var(--bg);
    }
    .stApp, [data-testid="stMarkdownContainer"] {
        font-size: 16.5px; line-height: 1.62; color: var(--text);
    }
    /* Paragraph rhythm: Times needs the extra leading far more than a sans
       does, and long advisory/explanatory passages are common in this app. */
    [data-testid="stMarkdownContainer"] p { line-height: 1.66; margin-bottom: 0.7rem; }

    section[data-testid="stSidebar"] { background: rgba(6,9,17,0.7); border-right: 1px solid var(--line-soft); }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(77,159,255,0.18); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(77,159,255,0.32); }

    /* ---- LabCD-matching top bar ---- */
    .lcd-topbar { display:flex; align-items:center; justify-content:space-between;
        padding: 0.85rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; border-bottom: 1px solid var(--line); }
    .lcd-brand { display:flex; align-items:center; gap:0.75rem; }
    .lcd-logo { width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
        background: var(--accent-dim); border: 1px solid rgba(77,159,255,0.22); }
    .lcd-brand-text .lcd-title { color:var(--text); font-size:var(--t-lead); font-weight:700; line-height:1.25; }
    .lcd-brand-text .lcd-subtitle { color:var(--text-3); font-size:var(--t-micro); line-height:1.3;
        letter-spacing:0.3px; }
    .lcd-nav { display:flex; align-items:center; gap:1.3rem; flex-wrap:wrap; }
    .lcd-nav-item { color:var(--text-2); font-size:var(--t-small); font-weight:500; white-space:nowrap;
        display:inline-flex; align-items:center; gap:0.4rem; }
    .lcd-nav-item svg { width:15px; height:15px; stroke: currentColor; flex-shrink:0; }
    .lcd-nav-item svg polygon[fill] { fill: currentColor; }
    .lcd-logo svg { width:20px; height:20px; stroke: currentColor; color:#8fc4ff; }
    .lcd-nav-pill { background: var(--accent); color:#08111f !important; padding:0.38rem 0.9rem;
        border-radius:8px; font-size:var(--t-small); font-weight:700; white-space:nowrap; }
    .lcd-nav-outline { border:1px solid var(--line-firm); color:var(--text-2); padding:0.34rem 0.8rem;
        border-radius:8px; font-size:var(--t-small); white-space:nowrap; }
    .lcd-avatar { width:26px; height:26px; border-radius:50%; background:linear-gradient(135deg,#818cf8,#4d9fff);
        display:inline-flex; align-items:center; justify-content:center; font-size:var(--t-micro);
        font-weight:700; color:#0a0d13; }

    /* ---- title card with icon-square ---- */
    .lcd-title-card { display:flex; align-items:flex-start; gap:1rem; background: var(--surface-1);
        border: 1px solid var(--line); border-radius: var(--radius); padding: 1.25rem 1.45rem; margin-bottom: 1rem; }
    .lcd-icon-square { min-width:44px; height:44px; border-radius:10px; background: var(--accent-dim);
        display:flex; align-items:center; justify-content:center; color:#5b9dff; }
    /* Icons dropped in here are defined with fill="none" and no stroke of
       their own (see LCD_ICON_FOLDER and friends), so the stroke has to be
       wired to the square's colour the same way .lcd-nav-item, .lcd-logo and
       .lcd-step-circle already do -- without this the icon paints nothing and
       the card shows an empty blue square. */
    .lcd-icon-square svg { width:22px; height:22px; stroke: currentColor; }
    .lcd-title-card .lcd-h { color:var(--text); font-size:var(--t-h1); font-weight:700;
        margin:0 0 0.25rem 0; line-height:1.25; }
    .lcd-title-card .lcd-sub { color:var(--text-3); font-size:var(--t-body); margin:0; line-height:1.5; }

    /* ---- stepper ---- */
    .lcd-stepper { display:flex; align-items:flex-start; justify-content:space-between; background: var(--surface-1);
        border: 1px solid var(--line); border-radius: var(--radius); padding: 1.5rem 2rem; margin-bottom: 1rem;
        position:relative; }
    .lcd-step { display:flex; flex-direction:column; align-items:center; text-align:center; flex:1;
        position:relative; z-index:2; }
    /* Each step draws its OWN connector segment running from its circle's
       right edge to the next step's circle's left edge -- deliberately NOT
       one continuous line with circles layered on top, which was still
       faintly visible crossing through the "done" circles (their fill is a
       semi-transparent blue, so a line behind them showed through). A segment
       that simply stops at calc(50% + 24px) can never render under a circle
       in the first place. */
    .lcd-step:not(:last-child)::after {
        content:""; position:absolute; top:24px; left:calc(50% + 24px); width:calc(100% - 48px); height:2px;
        background: var(--line-firm); z-index:1; border-radius:2px;
    }
    .lcd-step.done:not(:last-child)::after { background: var(--accent); }
    .lcd-step-circle { width:48px; height:48px; border-radius:50%; display:flex; align-items:center;
        justify-content:center; margin-bottom:0.65rem; border:1.5px solid var(--line-firm);
        background:#0f1219; color:var(--text-4);
        transition: border-color 0.25s ease, background 0.25s ease, color 0.25s ease; }
    .lcd-step-circle svg { width:22px; height:22px; stroke: currentColor; }
    .lcd-step-circle svg circle[fill], .lcd-step-circle svg path[fill] { fill: currentColor; }
    .lcd-step.done .lcd-step-circle { border-color:var(--accent); color:var(--accent); background:#12233a; }
    .lcd-step.active .lcd-step-circle { border-color:var(--accent); color:#08111f; background:var(--accent);
        box-shadow: 0 0 0 4px rgba(77,159,255,0.16); }
    .lcd-step-label { color:var(--text-2); font-size:var(--t-small); font-weight:700; }
    .lcd-step.active .lcd-step-label, .lcd-step.done .lcd-step-label { color:var(--text); }
    .lcd-step-sub { color:var(--text-4); font-size:var(--t-micro); margin-top:0.2rem; }

    /* ---- collapsible "Advanced Settings" look ---- */
    .lcd-advanced [data-testid="stExpander"] { border: 1px solid var(--line) !important;
        border-radius: var(--radius) !important; background: var(--surface-1) !important; }

    /* ---- native file_uploader restyled to look like the LabCD dropzone ---- */
    [data-testid="stFileUploaderDropzone"] { background: var(--surface-1) !important;
        border: 1.5px dashed var(--line-firm) !important; border-radius: var(--radius) !important;
        padding: 2.6rem 0 1.6rem 0 !important; position: relative !important; min-height: 120px !important;
        transition: border-color 0.2s ease, background 0.2s ease !important; }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(77,159,255,0.45) !important;
        background: rgba(77,159,255,0.03) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] svg { display:none; }
    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%235b9dff%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M12%203v12%22/%3E%3Cpath%20d%3D%22M7%208l5-5%205%205%22/%3E%3Cpath%20d%3D%22M4%2017v2a2%202%200%200%200%202%202h12a2%202%200%200%200%202-2v-2%22/%3E%3C/svg%3E");
        display:block; position:absolute; top:1.1rem; left:50%; transform:translateX(-50%);
        width:26px; height:26px; padding:11px; box-sizing:content-box;
        background:var(--accent-dim); border-radius:50%;
    }

    /* ---- cards ----
       One surface, one border, one shadow. The previous version stacked a
       14px backdrop blur, a gradient, an inset highlight, a drop shadow and a
       translateY hover on the same element; several of those at once is what
       read as busy rather than polished. */
    .glass-card, .metric-card {
        background: var(--surface-1);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--lift);
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    .glass-card { padding: 1.15rem 1.4rem; margin-bottom: 0.8rem; }
    .glass-card:hover { border-color: rgba(77,159,255,0.20); }
    .metric-card { padding: 0.95rem 1.1rem; text-align: center; }
    .metric-card:hover { border-color: rgba(77,159,255,0.22); background: var(--surface-2); }
    /* Letter-spaced uppercase is where a serif reads as deliberate rather
       than as a default -- these labels are the app's typographic signature. */
    .metric-card .label { font-size: var(--t-micro); color: var(--text-3); text-transform: uppercase;
        letter-spacing: 1.6px; }
    /* Numbers go monospace: Times has no tabular figures, so a column of
       metric values set in it visibly fails to line up. */
    .metric-card .value { font-family: var(--mono); font-size: 1.35rem; font-weight: 700;
        color: var(--text); letter-spacing: -0.3px; margin-top: 0.25rem; }
    .value-cyan { color: var(--accent); } .value-purple { color: var(--accent-2); }
    .value-yellow { color: var(--warn); } .value-green { color: var(--ok); } .value-red { color: var(--bad); }
    .status-ready { color: var(--accent); font-weight: 700; }
    .status-running { color: var(--warn); font-weight: 700; animation: pulse 1.8s ease-in-out infinite; }
    .status-failed { color: var(--bad); font-weight: 700; }
    @keyframes pulse { 0% {opacity:1;} 50% {opacity:0.55;} 100% {opacity:1;} }

    .progress-container { background: var(--surface-1); border-radius: var(--radius-sm); padding: 0.6rem 1rem;
        border: 1px solid var(--line-soft); }
    .progress-bar-bg { width:100%; height:5px; background: rgba(255,255,255,0.06); border-radius:3px;
        overflow:hidden; margin-top:0.45rem; }
    .progress-bar-fill { height:100%; background: linear-gradient(90deg,var(--accent),var(--accent-2));
        border-radius:3px; transition: width 0.4s cubic-bezier(0.2,0.8,0.2,1); }

    /* ---- logs and reasoning: monospace wells ---- */
    .log-container, .reasoning-container { background: var(--sunken); border-radius: var(--radius);
        padding: 0.7rem; max-height: 420px; overflow-y: auto; border: 1px solid var(--line-soft); }
    .log-entry { padding: 0.28rem 0.6rem; font-family: var(--mono); font-size: 0.76rem;
        color: var(--text-2); border-radius: 4px; line-height: 1.5; }
    .log-entry:hover { background: rgba(77,159,255,0.06); }
    .log-time { color: var(--text-4); } .log-node { color: var(--accent); font-weight: 700; }
    .log-metric { color: var(--warn); } .log-error { color: var(--bad); } .log-success { color: var(--ok); }
    .reasoning-entry { background: var(--surface-1); border-radius: var(--radius-sm); padding: 0.75rem 0.95rem;
        margin-bottom: 0.5rem; border-left: 3px solid rgba(77,159,255,0.35);
        transition: border-color 0.2s ease; }
    .reasoning-entry:hover { border-left-color: var(--accent); }
    .reasoning-entry .r-header { display:flex; justify-content:space-between; margin-bottom:0.35rem;
        font-size:var(--t-micro); }
    .reasoning-entry .r-node { font-weight:700; letter-spacing:1.4px; text-transform:uppercase; }
    .reasoning-entry .r-time { color: var(--text-4); font-family: var(--mono); }
    .reasoning-entry .r-text { color: var(--text-2); font-size: var(--t-small); line-height: 1.6;
        white-space: pre-wrap; }
    /* Agent identity colours, kept inside the one accent family. */
    .r-node-scenarist { color: var(--accent-2); } .r-node-actor { color: var(--accent); }
    .r-node-critic { color: var(--info); } .r-node-terminator { color: #7aa7f5; }
    .r-node-juror { color: var(--bad); } .r-node-evaluator { color: var(--ok); }

    /* ---- buttons ----
       No lift-on-hover: with a fragment re-rendering during a run, buttons
       that move under the cursor read as jitter. Colour carries the state. */
    .stButton button { background: var(--surface-2) !important;
        color: var(--text) !important; border: 1px solid var(--line) !important;
        border-radius: var(--radius-sm) !important; font-weight: 600 !important;
        box-shadow: var(--lift) !important;
        transition: background 0.16s ease, border-color 0.16s ease !important; }
    .stButton button:hover { background: var(--surface-3) !important;
        border-color: rgba(77,159,255,0.40) !important; }
    button[kind="primary"] { background: var(--accent) !important; color: #08111f !important;
        border: 1px solid var(--accent) !important; font-weight: 700 !important; }
    button[kind="primary"]:hover { background: #6bb0ff !important; border-color: #6bb0ff !important; }

    hr { border:none !important; height:1px !important;
        background: var(--line) !important; margin: 1.4rem 0 !important; }
    .glow-text { color: var(--accent); font-weight: 700; }
    .header-glass { background: var(--surface-1); border: 1px solid var(--line);
        border-radius: var(--radius); padding: 1.15rem 1.4rem; margin-bottom: 1.2rem;
        box-shadow: var(--lift); }
    .subheader { color: var(--text-3); font-size: var(--t-micro); font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.6px;
        margin: 1.5rem 0 0.6rem 0; padding-bottom: 0.45rem; border-bottom: 1px solid var(--line-soft); }
    .llm-badge { display:inline-block; background: var(--accent-dim); border: 1px solid rgba(77,159,255,0.28);
        color: var(--accent); border-radius: 999px; padding: 0.18rem 0.7rem; font-size: var(--t-micro);
        font-weight: 700; letter-spacing: 0.6px; }
    .fail-badge { display:inline-block; background: rgba(242,97,122,0.10); border: 1px solid rgba(242,97,122,0.30);
        color: var(--bad); border-radius: 999px; padding: 0.18rem 0.7rem; font-size: var(--t-micro);
        font-weight: 700; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color: var(--text-3); font-weight: 600; border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem; font-size: var(--t-body); }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(77,159,255,0.05); }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

    /* ---- Diagnostics (7th) and Manual Simulation (8th, only present once a
       run has stopped/finished) are the two "something needs you" tabs, not
       read-only result views like the six before them. `margin-left:auto` on
       the 7th pushes it AND everything after it to the right edge, so they
       read as their own group; their red/green label color comes from the
       :red[...] / :green[...] markdown in the tab labels themselves (see
       render_full_tuning_ui), which survives the aria-selected rule above
       because an inline span color beats an inherited one. ---- */
    .stTabs [data-baseweb="tab-list"] > [data-baseweb="tab"]:nth-of-type(7) {
        margin-left: auto; border-left: 1px solid var(--line); }

    div[data-testid="stMetric"] { background: var(--surface-1); border: 1px solid var(--line-soft);
        border-radius: var(--radius-sm); padding: 0.7rem 0.9rem; box-shadow: var(--lift); }
    [data-testid="stMetricValue"] { font-family: var(--mono) !important; }

    .dataframe { background: var(--surface-1) !important; border-radius: var(--radius) !important;
        border: 1px solid var(--line-soft) !important; }

    /* ---- Streamlit's own alerts, brought into the card system ----
       Left as defaults they were fully-saturated green/blue/red blocks that
       sat noticeably outside the rest of the palette. */
    [data-testid="stAlert"] { border-radius: var(--radius) !important; border: 1px solid var(--line) !important;
        box-shadow: var(--lift) !important; font-size: var(--t-body) !important; }
    [data-testid="stAlertContentSuccess"] { background: rgba(61,220,151,0.09) !important; }
    [data-testid="stAlertContentInfo"]    { background: rgba(77,159,255,0.09) !important; }
    [data-testid="stAlertContentWarning"] { background: rgba(232,177,61,0.09) !important; }
    [data-testid="stAlertContentError"]   { background: rgba(242,97,122,0.09) !important; }

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

    /* ---- site-wide font ----
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
    /* Heading scale. Streamlit's defaults are sized for a sans; in Times the
       same numbers read oversized and loose, so they are pulled in and given
       tighter leading. */
    h1 { font-size: var(--t-h1) !important; line-height: 1.25 !important; font-weight: 700 !important; }
    h2 { font-size: var(--t-h2) !important; line-height: 1.3 !important;  font-weight: 700 !important; }
    h3 { font-size: var(--t-lead) !important; line-height: 1.35 !important; font-weight: 700 !important; }
    code, pre, .stCode, [data-testid="stCodeBlock"] {
        font-family: var(--mono) !important; font-size: 0.84rem !important;
    }
    /* Widget labels sit one step below body text so a form reads as
       "value first, label second".
       Deliberately NOT uppercase + letter-spaced, even though that treatment
       is used for the standalone section labels above: it works on a two-word
       tag, but this app's widget labels are full phrases ("Min. turns before
       auto-complete", "Exploration intensity"), and uppercasing a phrase-length
       serif label costs more in scanability than it gains in style. */
    [data-testid="stWidgetLabel"] p {
        font-size: var(--t-small) !important;
        color: var(--text-3) !important;
        font-weight: 600 !important;
    }
    /* Reinforce on specific, high-confidence Streamlit text containers --
       these render only user-facing text content, never an icon ligature,
       so !important here is safe and fills in any gap plain inheritance
       might leave against Streamlit's own more-specific internal rules. */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"],
    [data-testid="stText"], [data-testid="stDataFrame"], [data-testid="stTable"],
    [data-testid="stMetricLabel"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    /* Data is the exception to the serif: monospace has tabular figures, the
       serif does not, so numeric columns only line up in the mono face. */
    [data-testid="stMetricValue"], [data-testid="stDataFrame"] [role="gridcell"],
    .metric-card .value, .log-entry, .r-time {
        font-family: var(--mono) !important;
    }
    [data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
    .material-symbols-sharp, .material-icons, .material-icons-outlined,
    .material-icons-round, .material-icons-sharp {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }

    /* ---- sliders ---- */
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="progressbar"] {
        background: var(--accent) !important;
        height: 5px !important; border-radius: 4px;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        width: 20px !important; height: 20px !important;
        background: radial-gradient(circle at 35% 30%, #ffffff, #8fc4ff 55%, var(--accent)) !important;
        box-shadow: 0 0 0 4px rgba(77,159,255,0.16), 0 2px 8px rgba(0,0,0,0.5) !important;
        border: none !important; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stSlider"] div[role="slider"]:hover,
    div[data-testid="stSlider"] div[role="slider"]:focus {
        transform: scale(1.12);
        box-shadow: 0 0 0 7px rgba(77,159,255,0.26), 0 2px 10px rgba(0,0,0,0.6) !important;
    }

    /* ---- card-selector: a grid of clickable cards standing in for a
       dropdown. Built entirely from st.button (full control, no BaseWeb
       internals to fight). The active card is marked via kind="primary",
       set programmatically in render_card_selector -- Streamlit doesn't nest
       widgets created after a raw st.markdown() div as its DOM children, so a
       parent-wrapper + nth-child approach can't reliably target the right
       card; styling the button's OWN kind attribute can. ---- */
    /* Card-selector buttons only. This used to read
           ...button[kind="secondary"].card-btn, div[data-testid="stButton"] button
       where the second member of the selector list carried no qualifier, so it
       matched EVERY button in the app and gave all of them an 84px floor plus
       left-aligned pre-wrap text -- which is why ordinary buttons like "Reset"
       rendered as oversized blocks with wrapped labels. (The `.card-btn` class
       in the old first member was never applied to anything.) render_card_selector
       keys its buttons "cardsel_*", and Streamlit puts that key on the element
       container as `st-key-<key>`, which gives the rule a real hook. */
    [class*="st-key-cardsel_"] div[data-testid="stButton"] button {
        min-height: 84px; border-radius: var(--radius);
        text-align: left; white-space: pre-wrap; line-height: 1.4;
    }
    .card-title { font-weight: 700; font-size: var(--t-body); color: inherit; margin-bottom: 2px; display:block; }
    .card-desc { font-weight: 400; font-size: var(--t-small); opacity: 0.85; display:block; }

    .weight-row { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.4rem; }
    .weight-label { width:110px; flex-shrink:0; font-size:var(--t-small); color:var(--text-3); text-align:right; }
    .weight-value { width:52px; flex-shrink:0; font-family:var(--mono); font-size:var(--t-small);
        color:var(--accent); font-weight:700; text-align:left; }
</style>"""

st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


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


def _perturbable_param_names(dyn) -> list:
    """Exactly which of a plugin's parameters Level 3 will perturb -- the
    same selection scenario_presets.perturb_physical_parameters makes
    (every OTHER numeric parameter, in sorted-name order), mirrored here so
    the Configure step can name them BEFORE a run instead of only after."""
    if dyn is None or not getattr(dyn, "params", None):
        return []
    numeric = sorted(
        k for k, v in dyn.params.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    return numeric[::2]


def _param_uncertainty_formula(dyn, keys: list, max_fraction: float) -> str:
    """The Level 3 mismatch as substituted math rather than prose: each
    parameter's own current value, the bound its random boost is drawn
    from, and the resulting worst-case value."""
    rows = []
    for k in keys[:8]:  # keep the block readable on a plugin with many params
        value = float(dyn.params[k])
        rows.append(
            rf"{_tex_var(k)} &= {value:.4g} \times (1 + \delta_{{{_tex_var(k)}}}), \quad "
            rf"\delta_{{{_tex_var(k)}}} \sim U(0,\ {max_fraction:.2f}) \;\Rightarrow\; "
            rf"\text{{up to }} \mathbf{{{value * (1 + max_fraction):.4g}}}"
        )
    if len(keys) > 8:
        rows.append(rf"&\text{{... and {len(keys) - 8} more}}")
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def render_perturbed_params_formulas(perturbed: dict):
    """What the plant ACTUALLY became for this run -- the realized draw per
    parameter, as substituted formulas. The Configure step can only show the
    bound (the draw happens once the run starts, seeded from the run's own
    random_seed), so this is the other half of that picture."""
    if not perturbed:
        return
    rows = []
    for name, (old, new) in list(perturbed.items())[:10]:
        pct = (new / old - 1) * 100 if old else 0.0
        rows.append(
            rf"{_tex_var(name)} &= {old:.4g} \times (1 + {pct / 100:.4f}) = "
            rf"\mathbf{{{new:.4g}}} \quad ({pct:+.1f}\%)"
        )
    st.latex(r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}")


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
            # The "cardsel_" prefix is what the CSS hooks onto (Streamlit puts
            # the widget key on the container as `st-key-<key>`), so that the
            # tall left-aligned card styling lands on these buttons and no
            # others. Buttons hold no persisted state, so the key is free to
            # change; the selection itself lives in `state_key` above.
            if st.button(label, key=f"cardsel_{key}_{i}", width="stretch",
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
    "authored_trajectory": None, "traj_author_chat": [],
    "derivative_pairs": [], "suggested_dt": None, "suggested_Q": None, "suggested_R": None, "setup_notes": [],
    "qr_diagnostics": None, "setup_panel_seen": False, "upload_panel_seen": False, "launch_step": 1,
    "last_outputs": {}, "stop_requested": False, "graph_iterator": None, "stopped_by_user": False, "run_error": None,
    # Why the run ended: None | "max_iterations" | "recursion_limit" | "terminator".
    "run_stop_reason": None,
    "report_pdf_bytes": None, "report_pdf_name": None, "run_perturbed_params": {},
    "ui_snapshot_html": None, "ui_snapshot_name": None,
    "upload_stage": None, "upload_review_code": "", "upload_review_filename": "", "upload_fix_result": None,
    "dynamics_source_code": None, "export_script_text": None, "export_script_name": None,
    "animation_gif_bytes": None, "animation_gif_name": None, "animation_description": None,
    "animation_render_note": None, "open_loop_result": None, "manual_best_iteration": None,
    "diagnostics_report": None, "diagnostics_chat_history": [], "token_tracker": None,
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
        st.session_state.upload_panel_seen = False  # show the loaded-file confirmation once for this new upload
        st.session_state.launch_step = 1  # start the Launch MPC wizard back at step 1
        # A trajectory conversation is about the PREVIOUS system's states and
        # derivative pairs -- carrying it into a new file would have the agent
        # revising a file written for a different state vector.
        st.session_state.authored_trajectory = None
        st.session_state.traj_author_chat = []
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
    the observed range (max-min, i.e. exactly what 1/range^2 uses) marked as
    an explicit bracket + numeric label, not just a shaded band in the title:
    the point of this panel is to make "range" something you can SEE, not
    just a number to take on faith."""
    traj = diagnostics["trajectory"]
    probe_dt = diagnostics["probe_dt"]
    ranges = diagnostics["ranges"]
    T = traj.shape[0]
    t = np.arange(T) * probe_dt
    n_states = traj.shape[1]

    palette = ["#4d9fff", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#22d3ee", "#fb7185", "#60a5fa",
               "#818cf8", "#38bdf8", "#fbbf24", "#a3e635"]

    # Never wider than the number of states: a 2-state plant used to be laid
    # out on a fixed 3-wide grid, so it rendered two big panels plus an empty
    # cell. The per-panel size is deliberately small -- this is a diagnostic
    # thumbnail showing the shape and range of each response, not a figure
    # anyone reads values off.
    ncols = max(1, min(3, n_states))
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 1.9 * nrows), squeeze=False)
    fig.patch.set_facecolor("#0a0e1a")
    t_span = max(t[-1] - t[0], 1e-9)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= n_states:
            ax.axis("off")
            continue
        color = palette[i % len(palette)]
        y = traj[:, i]
        y_min, y_max = float(y.min()), float(y.max())
        y_span = max(y_max - y_min, 1e-9)

        ax.plot(t, y, color=color, linewidth=2.0, zorder=3)
        ax.axhspan(y_min, y_max, color=color, alpha=0.10, zorder=1)
        ax.axhline(y_min, color=color, linewidth=0.8, linestyle=(0, (3, 2)), alpha=0.6, zorder=2)
        ax.axhline(y_max, color=color, linewidth=0.8, linestyle=(0, (3, 2)), alpha=0.6, zorder=2)
        # Equilibrium starting point, marked -- the probe's own origin.
        ax.plot(t[0], y[0], marker="o", color="#0a0e1a", markersize=4.5,
                markeredgecolor=color, markeredgewidth=1.4, zorder=4)

        # A bracket + exact numeric range at the right edge -- the same
        # measurement Bryson's rule uses below, made visible instead of
        # only ever appearing as a number in that formula.
        x_bracket = t[-1] + t_span * 0.10
        ax.annotate(
            "", xy=(x_bracket, y_max), xytext=(x_bracket, y_min),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.1, shrinkA=0, shrinkB=0),
        )
        ax.text(x_bracket + t_span * 0.05, (y_min + y_max) / 2, f"{ranges[i]:.3g}",
                color=color, fontsize=7, fontweight="bold", va="center", ha="left")

        name = state_names[i] if i < len(state_names) else f"x{i}"
        ax.set_title(name, color="#c7d2e8", fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlim(t[0] - t_span * 0.02, t[-1] + t_span * 0.34)
        ax.set_ylim(y_min - y_span * 0.22, y_max + y_span * 0.22)
        ax.tick_params(labelsize=6)
        if r == nrows - 1:
            ax.set_xlabel("t (s)", color="#5a6a8a", fontsize=6.5, labelpad=2)
        _style_ax(ax)

    plt.tight_layout()
    return fig


def plot_input_step_probe(diagnostics: dict, input_names: list):
    """Visualizes the constant open-loop input step (equilibrium +/- step_mag)
    the probe above actually applied per input -- exactly what 1/step^2
    uses for R. There is no time trajectory for this (it's a fixed offset
    from equilibrium, not a response), so this draws as a +/- bracket per
    input rather than a line: the same "make the measurement visible" idea
    as the range brackets above, applied to the one number that otherwise
    only ever appears inside the Bryson's-rule formula further down."""
    step_mag = diagnostics["step_mag"]
    n_inputs = len(step_mag)

    palette = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#60a5fa",
               "#fb7185", "#818cf8", "#fbbf24", "#38bdf8", "#4d9fff", "#a3e635"]

    fig, ax = plt.subplots(figsize=(3.0, 0.6 * n_inputs + 0.55))
    fig.patch.set_facecolor("#0a0e1a")
    max_step = max(step_mag) if step_mag else 1.0

    for j, s in enumerate(step_mag):
        color = palette[j % len(palette)]
        y = n_inputs - 1 - j  # top-to-bottom in declaration order
        ax.plot([-s, s], [y, y], color=color, linewidth=3.0, solid_capstyle="round", zorder=3)
        for x in (-s, s):
            ax.plot([x, x], [y - 0.16, y + 0.16], color=color, linewidth=1.4, zorder=3)
        ax.plot(0, y, marker="o", color="#0a0e1a", markersize=5.5,
                markeredgecolor=color, markeredgewidth=1.6, zorder=4)
        name = input_names[j] if j < len(input_names) else f"u{j}"
        ax.text(s + max_step * 0.14, y, f"±{s:.3g}", color=color, fontsize=7.5,
                fontweight="bold", va="center", ha="left")
        ax.text(0, y + 0.34, name, color="#c7d2e8", fontsize=7.5, fontweight="bold",
                va="bottom", ha="center")

    ax.axvline(0, color="#3a4a6a", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlim(-max_step * 1.35, max_step * 1.55)
    ax.set_ylim(-0.55, n_inputs - 1 + 0.55)
    ax.set_yticks([])
    ax.set_xlabel("input offset from equilibrium", color="#5a6a8a", fontsize=6.5, labelpad=3)
    ax.tick_params(axis="x", labelsize=6, colors="#3a4a6a")
    ax.set_facecolor("#0a0e1a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

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
    # Belt-and-braces against a caller passing a time axis of a different
    # length than the reference (a custom trajectory file's row count is
    # entirely up to whoever wrote it) -- plot the overlap rather than
    # raising out of a preview.
    n_common = min(len(times), len(reference))
    times, reference = times[:n_common], reference[:n_common]
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

    # A custom trajectory function decides its own row count -- some return
    # int(T/dt) samples, others int(T/dt)+1 (both endpoints included). The
    # time axis has to come from what actually came back, not from one
    # assumed convention: hardcoding int(T/dt) here is what raised
    # "x and y must have same first dimension, but have shapes (400,) and
    # (401,)" the first time an agent-written trajectory file was previewed.
    # Row k is the reference at time k*dt either way (that's how
    # run_closed_loop indexes ref_full), so deriving the axis from the row
    # count is correct for both.
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or reference.shape[0] == 0:
        st.warning(f"Couldn't preview this trajectory: expected a 2-D (n_steps, n_states) array, "
                   f"got shape {reference.shape}.")
        return
    times = np.arange(reference.shape[0]) * preview_dt
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
        # Raw mean-squared input, in the plant's own units -- deliberately NOT
        # normalized against the actuator bound / worst-so-far any more. The
        # normalized version read as a clean "0-1ish" number but hid the
        # actual magnitude behind a scale that changed meaning depending on
        # whether input bounds happened to be declared, which made the value
        # harder to compare across runs rather than easier.
        st.metric("Control Effort", f"{fmt_num(last['effort'])}", delta=f"{improvement.get('Control_Effort', 0.0):.1f}%",
                  help="Mean squared control input, in the plant's own units -- a proxy for actuator energy/wear.")

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
    st.caption("Runs one closed-loop simulation with parameters you choose directly -- no Agents, "
               "no LLM calls, just the MPC controller. Useful for sanity-checking a parameter set "
               "by hand, independent of the tuning loop. If you just stopped or finished a tuning "
               "run, the fields below are pre-filled with its last parameters.")

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
        # A number_input rather than a slider, and matching the Open Loop
        # Test's range: a slider capped at 30 s cannot show even one cycle of
        # a slow system (the Reaction Wheel reference sweeps with a ~126 s
        # period), and its 2 s floor blocks the opposite case -- zooming in on
        # the first fraction of a second of a stiff plant like the
        # electro-hydraulic servo.
        m_sim_time = st.number_input(
            "Simulation Time (s)", min_value=0.1, max_value=600.0,
            value=float(st.session_state.get("manual_sim_time", 10.0)),
            step=0.5, format="%.2f", key="manual_sim_time",
            help="How long to simulate for. Long windows on a fine dt_mpc mean many MPC solves -- "
                 "if a run feels slow, that product is why.",
        )
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
                else:
                    # Same degenerate case apply_scenario_level's Level 1/2 guard
                    # against (see scenario_presets.py) -- a plugin whose
                    # default_initial_state equals default_target (a common
                    # convention, especially for a nonzero setpoint, e.g. "start
                    # already at the operating RPM") would otherwise have exactly
                    # zero initial error here, silently making Overshoot show N/A
                    # with no way to tell why. Only applies when using the
                    # plugin's own default -- an explicit custom initial state
                    # above is left completely untouched.
                    dyn.config.default_initial_state = nudge_if_starts_at_target(
                        dyn, dyn.config.default_initial_state, dyn.config.default_target,
                    )

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
                st.metric("Control Effort", f"{fmt_num(m.control_effort)}",
                          help="Mean squared control input, in the plant's own units.")
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


def _apply_derivative_pairs(pairs):
    """Persist an edited pair list to every place that reads it.

    The detection result is written in three places when a file loads (see
    load/finalize_dynamics_load): session_state for the UI, and the live
    plugin's own config, which is what BaseDynamics uses to build a
    physically consistent reference (a state paired as a derivative gets
    cos where its partner gets sin, rather than an independent signal).
    An edit here has to reach both or the correction would be cosmetic.
    """
    st.session_state.derivative_pairs = pairs
    if st.session_state.get("dyn") is not None:
        st.session_state.dyn.config.derivative_pairs = pairs or None


def render_derivative_pairs_editor(state_names):
    """Shows the detected derivative pairs and lets them be corrected by hand.

    Detection (agents/dynamics_validator.py:detect_derivative_pairs) compares
    dxi/dt against xj numerically at random points, which is reliable for a
    textbook plant but can miss a pair -- or claim one -- on a system where two
    states happen to track each other over the sampled region. Since the pairs
    decide how per-state reference trajectories are built, a wrong pair is
    silently wrong rather than obviously wrong, so the detection is presented
    as a starting point the user can override instead of a fact.
    """
    pairs = st.session_state.get("derivative_pairs") or []
    if pairs:
        pair_strs = [f"**{state_names[j]}** = d(**{state_names[i]}**)/dt" for i, j in pairs
                     if i < len(state_names) and j < len(state_names)]
        st.markdown("**Derivative pairs detected** (verified numerically -- dxᵢ/dt ≡ xⱼ at many "
                    "random points): " + ", ".join(pair_strs))
    else:
        st.markdown("**Derivative pairs:** none detected.")

    if not state_names:
        return

    with st.expander("Edit derivative pairs", expanded=False):
        st.caption(
            "For each state, say which other state is its time derivative. The detector filled "
            "this in numerically; correct it here if it got a pair wrong or missed one. This is "
            "what lets a position state be tracked with a sine while its velocity state gets the "
            "matching cosine, instead of the two being driven independently."
        )
        NONE = "— none —"
        current = {i: j for i, j in pairs}
        rows = pd.DataFrame({
            "State": list(state_names),
            "Its derivative is": [
                state_names[current[i]] if i in current and current[i] < len(state_names) else NONE
                for i in range(len(state_names))
            ],
        })
        edited = st.data_editor(
            rows, hide_index=True, width="stretch", key="derivative_pairs_editor",
            column_config={
                "State": st.column_config.TextColumn(disabled=True),
                "Its derivative is": st.column_config.SelectboxColumn(options=[NONE] + list(state_names)),
            },
        )

        new_pairs, problems = [], []
        name_to_idx = {n: k for k, n in enumerate(state_names)}
        for i, choice in enumerate(edited["Its derivative is"]):
            if choice == NONE:
                continue
            j = name_to_idx.get(choice)
            if j is None:
                continue
            if j == i:
                problems.append(f"**{state_names[i]}** cannot be its own derivative.")
                continue
            new_pairs.append((i, j))

        # A state can only be one state's derivative -- two positions sharing a
        # velocity would make the reference builder ambiguous.
        seen = {}
        for i, j in new_pairs:
            if j in seen:
                problems.append(
                    f"**{state_names[j]}** is set as the derivative of both "
                    f"**{state_names[seen[j]]}** and **{state_names[i]}**."
                )
            seen[j] = i

        if problems:
            for p in problems:
                st.error(p, icon=":material/error:")
        elif st.button("Apply pairs", key="apply_derivative_pairs", type="primary"):
            _apply_derivative_pairs(new_pairs)
            st.success(f"Saved {len(new_pairs)} derivative pair(s).")
            st.rerun()


def _tex_var(name: str) -> str:
    """Escapes a state/input name for safe use inside a LaTeX \\text{} block.
    Names are free-form Python identifiers (e.g. "cart_pos"), and a bare
    underscore in LaTeX starts a subscript -- unescaped, "cart_pos" would
    silently render as "cart" with a stray "pos" floating below it instead
    of the literal name."""
    return name.replace("_", r"\_") if name else name


def _bryson_formula_block(names: list, values: list, measured: list, symbol: str) -> str:
    """One aligned LaTeX block with a real substituted equation per
    state/input -- e.g. ``Q_{\\text{theta}} = 1/0.704^2\\ (rescaled) =
    10.00`` -- instead of a single prose example sentence standing in for
    every variable. ``measured`` is the probe's own range_i (for Q) or
    step_j (for R); ``values`` is the final rescaled weight already in
    st.session_state.suggested_Q/R."""
    rows = r" \\ ".join(
        rf"{symbol}_{{\text{{{_tex_var(names[i] if i < len(names) else f'{symbol.lower()}{i}')}}}}}"
        rf" &= \dfrac{{1}}{{{measured[i]:.3g}^{{2}}}}\ (\text{{rescaled}}) = \mathbf{{{fmt_num(values[i])}}}"
        for i in range(len(values))
    )
    return r"\begin{aligned}" + rows + r"\end{aligned}"


def render_setup_agent_panel():
    """Graphical walkthrough of what the Initial Setup Agent did to THIS
    upload (agents/dynamics_validator.py:analyze_and_setup) -- shown once,
    prominently, right after a dynamics file is loaded and BEFORE the
    tuning-run state-flow diagram ever appears (that one only exists once
    "Run" is clicked, so the sequencing -- setup first, tuning-loop diagram
    second -- is automatic). Answers three questions visually: was the file
    OK as uploaded (or what got fixed), which states are derivative pairs,
    and where the suggested Q/R/dt numbers actually came from -- the last
    one as a real substituted formula per variable, not one example sentence
    standing in for all of them."""
    was_fixed = bool(st.session_state.get("fixed_dynamics_code"))
    summary = st.session_state.dynamics_summary
    state_names = summary.get("state_names", [])
    input_names = summary.get("input_names", [])
    diag = st.session_state.get("qr_diagnostics")
    Q, R = st.session_state.get("suggested_Q"), st.session_state.get("suggested_R")
    suggested_dt = st.session_state.get("suggested_dt")
    pairs = st.session_state.get("derivative_pairs") or []

    # ---- summary strip: the four things this whole panel explains, at a
    # glance, in the same metric-card language the results dashboard already
    # uses -- so this reads as part of the same product, not a bespoke form. ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="subheader" style="margin-top:0;">{LCD_ICON_WRENCH_SM} Initial Setup Agent \u2014 what it found for this file</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Validation</div>'
            f'<div class="value {"value-yellow" if was_fixed else "value-green"}" style="font-size:1rem;">'
            f'{"Auto-repaired" if was_fixed else "Valid as uploaded"}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Derivative pairs</div>'
            f'<div class="value">{len(pairs)}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(
            f'<div class="metric-card"><div class="label">dt_mpc (starting)</div>'
            f'<div class="value value-cyan">{f"{suggested_dt:.4g}s" if suggested_dt else "n/a"}</div></div>',
            unsafe_allow_html=True)
    with s4:
        st.markdown(
            f'<div class="metric-card"><div class="label">States / Inputs</div>'
            f'<div class="value">{len(state_names)} / {len(input_names)}</div></div>', unsafe_allow_html=True)

    if was_fixed:
        st.caption(st.session_state.get("fixed_dynamics_explanation", ""))
    else:
        st.caption("Structurally matched the dynamics standard on the first check -- no LLM repair needed.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- derivative pairs (detected, and editable) ----
    render_derivative_pairs_editor(state_names)

    # ---- step-response probe + Bryson's rule math ----
    if diag and Q:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader" style="margin-top:0;">Step-response probe</div>', unsafe_allow_html=True)
        st.caption("Open-loop, {:.0%} input step on top of equilibrium -- the bracketed span on each state's "
                   "trace, and each input's +/- bar, are exactly what the Q/R weights below are measured "
                   "from:".format(0.25))
        # Pinned to explicit pixel widths. "content" is not enough on its own:
        # Streamlit rasterizes the figure at a high dpi, so its natural size is
        # wider than the panel and it gets fitted to the full container width
        # regardless -- which is what made this diagnostic thumbnail dominate
        # the page. Scale with the number of columns so a 4-state plant still
        # gets readable panels, but cap it well short of full width.
        _probe_cols = max(1, min(3, len(state_names)))
        _state_px, _input_px = min(660, 240 * _probe_cols), 190
        col_state_probe, col_input_probe = st.columns([_state_px, _input_px])
        with col_state_probe:
            fig = plot_step_response_probe(diag, state_names)
            st.pyplot(fig, width=_state_px)
            plt.close(fig)
        with col_input_probe:
            fig_u = plot_input_step_probe(diag, input_names)
            st.pyplot(fig_u, width=_input_px)
            plt.close(fig_u)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader" style="margin-top:0;">Bryson\'s rule</div>', unsafe_allow_html=True)
        st.caption("Weight each variable by the inverse square of its own characteristic scale, so no state "
                   "or input dominates the cost just because of its units:")
        st.latex(r"Q_{ii} = \frac{1}{\text{range}_i^{\,2}} \cdot \frac{Q_{\max}}{\max_k(1/\text{range}_k^{\,2})} "
                 r"\qquad\qquad R_{jj} = \frac{1}{\text{step}_j^{\,2}} \cdot \frac{R_{\max}}{\max_k(1/\text{step}_k^{\,2})}")

        # Every state and every input gets its own substituted equation here
        # -- range_i / step_j as actually measured by the probe above, and
        # the resulting weight -- rather than one example sentence for
        # whichever single state happened to have the largest Q.
        qc, rc = st.columns(2)
        with qc:
            st.markdown(
                '<div style="color:var(--text-2); font-weight:600; font-size:0.85rem; margin-bottom:4px;">'
                'State weights (Q) \u2014 measured range per state</div>', unsafe_allow_html=True)
            st.latex(_bryson_formula_block(state_names, Q, diag["ranges"], "Q"))
        with rc:
            st.markdown(
                '<div style="color:var(--text-2); font-weight:600; font-size:0.85rem; margin-bottom:4px;">'
                'Input weights (R) \u2014 measured step per input</div>', unsafe_allow_html=True)
            st.latex(_bryson_formula_block(input_names, R, diag["step_mag"], "R"))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Step-response probe unavailable for this file (see setup notes below).")

    # ---- dt reasoning ----
    if suggested_dt:
        st.markdown(
            '<div class="glass-card" style="border-left:3px solid var(--accent);">'
            '<div class="subheader" style="margin-top:0;">Sample time</div>'
            f'<span style="font-family:var(--mono); font-weight:700; font-size:1.15rem; color:var(--accent);">'
            f'dt_mpc &asymp; {suggested_dt:.4g}s</span>'
            '<p style="margin:8px 0 0 0; color:var(--text-2); font-size:0.85rem; line-height:1.6;">'
            "The smaller of two estimates: (a) 1/15th of the fastest linearized time constant at "
            "equilibrium, and (b) 1/8th of the fastest state's step-response rise time. Used as the "
            "STARTING value; the Actor may periodically adjust it during the run.</p></div>",
            unsafe_allow_html=True,
        )


def _current_flow_entry(active_node: Optional[str], reasoning_entries: list) -> Optional[Dict[str, str]]:
    """Picks the single entry the box below the flow diagram shows right
    now -- automatically, no hover, no click. During a live run,
    ``active_node`` is the node that JUST executed (see run_one_step: the
    node's own history entry is appended before this diagram is re-rendered
    for that tick), so its own latest entry is shown. Once idle -- before a
    run starts, or after one ends -- there is no "current" node, so the
    most recent entry overall is shown instead, keeping the box meaningful
    rather than going blank the moment a run finishes.

    Returns a dict shaped exactly like what render_reasoning_panel builds
    per entry (node_label, css_class, time, body) -- deliberately: this box
    IS an Agent Reasoning entry, just always the current one, rendered with
    the identical .reasoning-entry markup so the two views never look like
    two different features. Returns None if there is nothing to show yet.
    """

    def _as_dict(text: str, time_str: str) -> Dict[str, str]:
        node_label = text.split("]")[0].lstrip("[") if text.startswith("[") else "INFO"
        body = text.split("]", 1)[1].strip() if "]" in text else text
        return {"node_label": node_label, "css_class": _reasoning_node_class(text), "time": time_str, "body": body}

    static_copy = {
        "evaluator": "No reasoning -- the Evaluator deterministically runs the closed-loop "
                     "simulation and computes metrics; nothing here is agent-generated text.",
    }
    if active_node in static_copy:
        return {"node_label": active_node.capitalize(), "css_class": "", "time": "",
                "body": static_copy[active_node]}

    if active_node:
        wanted_prefix = f"[{active_node.capitalize()}]"
        for entry in reversed(reasoning_entries or []):
            if entry.get("text", "").startswith(wanted_prefix):
                return _as_dict(entry["text"], entry["time"])
        return None  # this node hasn't produced an entry yet this run

    if reasoning_entries:
        last = reasoning_entries[-1]
        return _as_dict(last["text"], last["time"])

    return None


def render_agent_flow_diagram(active_node: Optional[str] = None, iteration: int = 0, last_decision: str = "",
                               reasoning_entries: Optional[list] = None):
    """A small Simulink-Stateflow-style live diagram of the tuning graph
    (Actor -> Evaluator -> Terminator -> {Critic|Juror}, with Critic feeding
    back to Actor and Juror -- now the run's mandatory final reviewer, not
    just an escalation handler -- feeding back to Actor OR ending the run).
    The node that just executed is highlighted
    with a glowing, pulsing border; everything else stays dim. Re-rendered
    (cheap -- it's just an SVG string) after every node during a live run,
    same pattern as the rest of the live-updating panels.

    A box below the diagram shows that CURRENT node's full output
    automatically -- see _current_flow_entry -- no hovering required: an
    earlier version needed a mouse over the node (first as a floating
    tooltip, then as a box revealed by CSS :hover), which meant the output
    was invisible by default and only one interaction away from being
    missed entirely, especially during an unattended live run. Rendered
    with the exact same .reasoning-entry markup the Agent Reasoning tab
    uses, so this is not a second, differently-styled feature -- it is the
    Agent Reasoning tab's current entry, surfaced where the diagram already
    has your attention.
    """
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

    st.markdown(f'<div class="flow-diagram-wrap">{svg}</div>', unsafe_allow_html=True)

    caption = f"Iteration {iteration}"
    if last_decision:
        caption += f"  &middot;  {last_decision}"
    st.caption(caption)

    # The current node's output, automatically -- reuses the Agent Reasoning
    # tab's OWN markup (.reasoning-container / .reasoning-entry / .r-node /
    # .r-text, all already defined in DASHBOARD_CSS) so this reads as that
    # same feature rather than a second, differently-styled one.
    current = _current_flow_entry(active_node, reasoning_entries or [])
    if current is None:
        body_html = (
            '<div class="reasoning-entry"><div class="r-text" style="font-style:italic; color:var(--text-4);">'
            "No agent activity yet -- run a tuning session to see output here.</div></div>"
        )
    else:
        time_html = f'<span class="r-time">{html_module.escape(current["time"])}</span>' if current["time"] else ""
        body_html = (
            '<div class="reasoning-entry">'
            f'<div class="r-header"><span class="r-node {current["css_class"]}">'
            f'{html_module.escape(current["node_label"])}</span>{time_html}</div>'
            f'<div class="r-text">{current["body"]}</div></div>'
        )
    st.markdown(f'<div class="reasoning-container" style="max-height:220px;">{body_html}</div>',
                unsafe_allow_html=True)


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
    (LCD_ICON_SLIDERS, "Configure", "Scenario & trajectory"),
    (LCD_ICON_ROCKET, "Tune", "Run the agents"),
]
# The index used to go 0 -> 2 -> 3, so "Setup Agent" was never the active
# step: loading a file jumped the stepper straight from Upload to Configure,
# even though the Setup Agent's findings are the thing on screen at that
# moment. It now rests on step 1 until those findings have been acknowledged
# (the same flag that collapses the panel), so the stepper matches what the
# page is actually showing. Likewise, the loaded-file confirmation (System
# card: filename, shape, Test Dynamics, Open Loop Test) is still about the
# thing you just uploaded, not the Setup Agent's analysis of it -- so the
# stepper stays on "Upload" (index 0) until THAT is acknowledged too.
if not st.session_state.dynamics_loaded:
    _lcd_step_index = 0
elif st.session_state.results_data or st.session_state.running:
    _lcd_step_index = 3
elif not st.session_state.upload_panel_seen:
    _lcd_step_index = 0
elif st.session_state.setup_notes and not st.session_state.setup_panel_seen:
    _lcd_step_index = 1
else:
    _lcd_step_index = 2
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

def render_loaded_system_panel():
    """What was loaded, and the two diagnostics you'd want before tuning it.

    This used to live inside the Setup section's "1 - System" tab, behind a
    second copy of the upload dropzone -- so the file's identity and shape were
    a tab-click away from the page you land on straight after uploading, and
    the tab re-offered an upload the user had just completed. The upload UI is
    gone (the landing page already owns that job) and what is left surfaces
    here instead: file, shape, the Setup Agent's repair download when there was
    one, and the two sanity checks (closed-loop Test Dynamics, and the
    no-controller Open Loop Test that used to be buried in the results area's
    Simulation tab, only reachable after a run had produced something).
    """
    summary = st.session_state.dynamics_summary
    with st.container(border=True):
        st.markdown(
            '<div style="color:#f1f3f7; font-weight:700; margin-bottom:0.6rem;">System</div>',
            unsafe_allow_html=True,
        )
        _c1, _c2 = st.columns([3, 2])
        with _c1:
            st.success(f"Loaded: {summary.get('source_file', 'Unknown')}")
        with _c2:
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

        if st.button("Test Dynamics", width="stretch", key="test_dynamics_landing",
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

        render_open_loop_test()


# Both of these describe the file you are about to tune. Once a run is going
# (or has produced results) they are just scroll between the user and the
# results, so they drop away -- the sidebar still carries the system identity.
_pre_flight = not st.session_state.results_data and not st.session_state.running

# ---------------------------------------------------------------------------
# UPLOAD CONFIRMATION -> SETUP AGENT -> CONFIGURE -- three actual steps, not
# one long scroll.
#
# These used to render one after another unconditionally: the loaded-file
# System panel, the Setup Agent findings, AND the Launch MPC / tabs section
# were all on screen at once the moment a file loaded -- and, because nothing
# ever stopped a rerun from falling through past whichever of these it had
# just rendered, the results dashboard below (Live Run / flow diagram /
# report buttons -- which only means anything once a run exists) rendered
# unconditionally too, showing up at the bottom of EVERY step's page,
# including this one, before a run had even started. Each step below is now
# a self-contained if-block that ends in st.stop(): render this step's
# content, offer the button that advances past it, and stop -- so a step's
# content, the next step's content, and the results dashboard can never all
# land on screen in the same rerun.
#
# If a file has no setup_notes at all (the deterministic scan found nothing
# to say), there is no Setup Agent step to show -- go straight to Configure,
# same fallback the stepper index above already uses.
# ---------------------------------------------------------------------------
_show_upload_confirm_step = (_pre_flight and st.session_state.dynamics_loaded
                              and not st.session_state.upload_panel_seen)
_show_setup_step = (_pre_flight and st.session_state.upload_panel_seen
                     and bool(st.session_state.setup_notes) and not st.session_state.setup_panel_seen)
_show_configure_step = (_pre_flight and st.session_state.upload_panel_seen
                         and (not st.session_state.setup_notes or st.session_state.setup_panel_seen))

if _show_upload_confirm_step:
    render_loaded_system_panel()

    _next_step_label = "Continue to Setup Agent →" if st.session_state.setup_notes else "Continue to Configure →"
    if st.button(_next_step_label, type="primary", width="stretch", key="continue_to_setup"):
        st.session_state.upload_panel_seen = True
        st.rerun()
    st.stop()  # never fall through into the Setup Agent step or the results dashboard

if _show_setup_step:
    if st.button("← Back to Upload", key="back_to_upload"):
        st.session_state.upload_panel_seen = False
        st.rerun()

    render_lcd_title_card(
        LCD_ICON_WRENCH_SM, "Setup Agent",
        "What the deterministic analysis found for this file -- validation, derivative "
        "structure, and where the suggested Q/R/dt numbers come from. No LLM involved.",
    )
    with st.expander("What it found for this file", expanded=True):
        render_setup_agent_panel()

    if st.button("Continue to Configure →", type="primary", width="stretch", key="continue_to_configure"):
        st.session_state.setup_panel_seen = True
        st.rerun()
    st.stop()  # never fall through into the Configure step or the results dashboard

if _show_configure_step:
    if st.session_state.setup_notes:
        # Only offered when there WAS a Setup Agent step to return to --
        # otherwise Configure comes straight after the Upload confirmation.
        if st.button("← Back to Setup Agent", key="back_to_setup_agent"):
            st.session_state.setup_panel_seen = False
            st.rerun()
    else:
        if st.button("← Back to Upload", key="back_to_upload_from_configure"):
            st.session_state.upload_panel_seen = False
            st.rerun()

    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.6rem; margin:0.4rem 0 1rem 0;">'
        '<div style="width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,#1b3a63,#122238); '
        'display:flex; align-items:center; justify-content:center; flex-shrink:0;">' + LCD_ICON_SLIDERS +
        '</div><div><div style="font-weight:700; font-size:1.05rem; color:#f1f3f7;">Launch MPC</div>'
        '<div style="font-size:0.82rem; color:#6a7a9a;">Three quick steps, then launch -- every default is sensible, edit only what you need to.</div>'
        '</div></div>', unsafe_allow_html=True,
    )
    # A genuine sequential wizard, not a free-roaming tab strip: a later
    # step's controls -- and Run -- don't exist on the page at all until the
    # step before it has been stepped past. With real st.tabs() every tab is
    # always clickable, so nothing stopped landing on tab 1 and hitting Run
    # without ever seeing what the other tabs configure.
    #
    # Each step's expander keeps RENDERING (collapsed) once you are past it,
    # so every variable it defines stays defined for the later steps and for
    # the Run handler at the bottom -- only steps AHEAD of the current one
    # are skipped, and Run only exists on the last one.
    _launch_step = int(st.session_state.launch_step)
    n_states_hint = st.session_state.dynamics_summary.get("n_states", 4) if st.session_state.dynamics_loaded else 4
    n_inputs_hint = st.session_state.dynamics_summary.get("n_inputs", 1) if st.session_state.dynamics_loaded else 1

    with st.expander(
        "1  ·  Initial State & Reference" + ("   ✓" if _launch_step > 1 else ""),
        expanded=_launch_step == 1,
    ):
        # Always visible and always editable -- no "enable this first"
        # checkbox in front of it. Pre-filled with the plugin's own declared
        # default_initial_state; it only counts as an OVERRIDE once you
        # actually change a value, so leaving it untouched still lets the
        # Scenario Level's own initial-state handling apply (see
        # scenario_presets.nudge_if_starts_at_target, which rescues the
        # degenerate "starts exactly at the target" case that otherwise makes
        # Overshoot unmeasurable).
        st.markdown('<div style="color:#f1f3f7; font-weight:700;">Initial state</div>', unsafe_allow_html=True)
        _plugin_default_x0 = (
            list(st.session_state.dyn.config.default_initial_state)
            if st.session_state.dynamics_loaded else []
        )
        init_state_text = st.text_input(
            f"Initial state -- {n_states_hint} comma-separated values",
            value=", ".join(f"{fmt_num(v)}" for v in _plugin_default_x0),
            key="configure_initial_state_text",
            help="Where every run starts from, one value per state, in the order shown in the sidebar's "
                 "'States' summary. Pre-filled from the dynamics file itself -- change any value and "
                 "yours is used instead (applied AFTER the Scenario Level preset, so it wins over it).",
        )
        custom_initial_state, initial_state_error = None, None
        try:
            _parsed_x0 = np.array([float(v.strip()) for v in init_state_text.split(",") if v.strip()])
            if _parsed_x0.size != n_states_hint:
                initial_state_error = f"Need exactly {n_states_hint} value(s), got {_parsed_x0.size}."
            else:
                custom_initial_state = _parsed_x0
        except ValueError:
            initial_state_error = "Must be comma-separated numbers."
        if initial_state_error:
            st.error(initial_state_error)
        use_custom_initial_state = (
            custom_initial_state is not None and bool(_plugin_default_x0)
            and not np.allclose(custom_initial_state, np.asarray(_plugin_default_x0, dtype=float))
        )

        st.divider()
        st.subheader("Trajectory")

        with st.expander("Custom Reference Trajectory (optional)", expanded=False):
            with st.expander("Trajectory File Standard (reference)", expanded=False):
                from backend_core.AgentMPC.agents.trajectory_validator import TRAJECTORY_STANDARD
                st.markdown(TRAJECTORY_STANDARD)

            # ---- describe it instead of writing it ----
            # The uploader below assumes you already have a trajectory file.
            # This covers knowing what you want the reference to do without
            # wanting to express it in NumPy. Whatever comes back is put
            # through the same validator the uploaded files go through.
            st.markdown("**Describe the trajectory you want**")
            st.caption(
                "Name the states and what each should follow -- e.g. \"theta1 sinusoidal and omega1 its "
                "cosine, amplitude 0.2, frequency 0.4 Hz\". States you don't mention stay at zero. "
                "After the first draft you can keep talking to it (\"make the amplitude 0.1\", \"give "
                "theta2 a pulse too\") and it revises the file it already wrote."
            )

            # Show exactly what structural knowledge the agent is handed, so
            # a wrong pairing is visible HERE rather than only discoverable
            # from a reference that turns out to be physically inconsistent.
            # These are the same pairs the Setup Agent detected (and that the
            # "Edit derivative pairs" editor there corrects).
            _traj_state_names = st.session_state.dynamics_summary.get("state_names", [])
            _traj_pairs = st.session_state.get("derivative_pairs") or []
            if _traj_pairs and _traj_state_names:
                _pair_text = ", ".join(
                    f"`{_traj_state_names[j]} = d({_traj_state_names[i]})/dt`"
                    for i, j in _traj_pairs
                    if i < len(_traj_state_names) and j < len(_traj_state_names)
                )
                st.caption(f"The agent is told the state order (`{', '.join(_traj_state_names)}`) and the "
                           f"detected derivative pairs -- {_pair_text} -- so a sinusoidal position gets its "
                           f"true derivative (amplitude x omega) on the paired velocity, not a same-amplitude "
                           f"cosine. Correct them in the Setup Agent step if any pair is wrong.")
            elif _traj_state_names:
                st.caption(f"The agent is told the state order (`{', '.join(_traj_state_names)}`). No "
                           f"derivative pairs were detected for this system, so it treats every state as "
                           f"independent unless you say otherwise.")

            for _turn in st.session_state.traj_author_chat:
                with st.chat_message(_turn["role"]):
                    st.write(_turn["content"])

            _has_draft = st.session_state.get("authored_trajectory") is not None
            # The key changes every turn so each message gets a FRESH (empty)
            # box: Streamlit forbids writing a widget's own session_state key
            # after that widget has been instantiated in the same run, so
            # clearing it after send is only possible by making it a new widget.
            _traj_request = st.text_area(
                "Trajectory description", key=f"traj_author_request_{len(st.session_state.traj_author_chat)}",
                height=80, label_visibility="collapsed",
                placeholder=("What should change? e.g. \"amplitude 0.1 instead\", \"add a pulse on theta2\""
                             if _has_draft else
                             "theta1 sinusoidal, omega1 its cosine, amplitude 0.2, frequency 0.4 Hz"),
            )
            _traj_cols = st.columns([3, 1]) if _has_draft else [st.container()]
            with _traj_cols[0]:
                # Deliberately NOT disabled on an empty box: a text_area only
                # commits its value on blur, so the click that blurs it is the
                # same click that would have to find the button already
                # enabled -- with `disabled=` the first click is swallowed and
                # the feature looks broken. Validate on press instead.
                _traj_go = st.button(
                    "Send revision" if _has_draft else "Write the trajectory file",
                    key="traj_author_go", type="primary", width="stretch",
                )
            if _has_draft:
                with _traj_cols[1]:
                    if st.button("Start over", key="traj_author_reset", width="stretch"):
                        st.session_state.authored_trajectory = None
                        st.session_state.traj_author_chat = []
                        st.rerun()

            if _traj_go:
                if not (_traj_request or "").strip():
                    st.warning("Describe what you want first (or what you'd like changed).")
                elif not LLM_READY:
                    st.error(f"This needs the LLM to be configured -- {LLM_INIT_ERROR}")
                else:
                    from backend_core.AgentMPC.agents.trajectory_author_agent import author_trajectory
                    _prev = st.session_state.get("authored_trajectory")
                    try:
                        with st.spinner("Revising and validating the trajectory file..." if _prev is not None
                                        else "Writing and validating the trajectory file..."):
                            _authored = author_trajectory(
                                _traj_request,
                                st.session_state.dynamics_summary,
                                derivative_pairs=st.session_state.get("derivative_pairs"),
                                tracker=st.session_state.get("token_tracker"),
                                conversation_history=list(st.session_state.traj_author_chat),
                                previous_code=(_prev.code if _prev is not None and _prev.code else None),
                            )
                        st.session_state.authored_trajectory = _authored
                        # The thread carries the EXPLANATIONS, not the code --
                        # the code itself is fed back separately as the file to
                        # revise, so the prompt doesn't grow by a full copy of
                        # every draft on every turn.
                        st.session_state.traj_author_chat = st.session_state.traj_author_chat + [
                            {"role": "user", "content": _traj_request},
                            {"role": "assistant", "content": _authored.explanation or "(no explanation returned)"},
                        ]
                        st.rerun()
                    except Exception as _exc:  # noqa: BLE001
                        st.error(f"The trajectory agent failed: {_exc}")

            _authored = st.session_state.get("authored_trajectory")
            if _authored is not None:
                if not _authored.valid:
                    st.error(f"Couldn't produce a loadable trajectory file: {_authored.error}")
                    if _authored.code:
                        with st.expander("Last draft (not loaded)", expanded=False):
                            st.code(_authored.code, language="python")
                else:
                    st.success("Trajectory file written and validated.")
                    if _authored.was_repaired:
                        st.caption("The first draft didn't load as written and was repaired automatically, "
                                   "so the explanation below describes the draft rather than the final file.")
                    st.info(_authored.explanation)
                    st.code(_authored.code, language="python")
                    _ta1, _ta2 = st.columns([1, 1])
                    with _ta1:
                        if st.button("Use this trajectory", key="traj_author_use", type="primary"):
                            from backend_core.AgentMPC.dynamics.trajectory_loader import TrajectoryLoader
                            import tempfile as _tempfile
                            with _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                                               encoding="utf-8") as _f:
                                _f.write(_authored.code)
                                _tpath = _f.name
                            try:
                                st.session_state.custom_trajectory_loader = TrajectoryLoader.load_from_path(_tpath)
                                st.session_state.trajectory_file_name = "agent_written_trajectory.py"
                                st.success("Loaded. Select 'Custom' as the trajectory type to use it.")
                                st.rerun()
                            except Exception as _exc:  # noqa: BLE001
                                st.error(f"Failed to load the written trajectory: {_exc}")
                    with _ta2:
                        st.download_button(
                            "Download .py", _authored.code, file_name="trajectory.py",
                            mime="text/x-python", key="traj_author_download",
                        )

            st.markdown("**Or upload a file you already have**")
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
        # Per-state reference override, off by default. Left off, the single
        # trajectory type above applies to the whole system and BaseDynamics
        # pairs each position state's sine with its velocity state's cosine
        # itself, from the derivative pairs the Setup Agent detected -- which
        # is what you want almost always. Turned ON, you get one row per
        # state and pick its reference independently, for the cases the
        # automatic pairing can't express (e.g. hold one state steady while
        # another tracks a wave). Doesn't apply to a Custom trajectory file:
        # that path builds the whole reference array itself (see
        # run_closed_loop's trajectory_mode == "custom" branch), so the
        # editor isn't offered there rather than being silently ignored.
        per_state_trajectory_modes = None
        _per_state_labels = {"reg": "Regulation", "sin": "Sinusoidal", "cos": "Cosinusoidal", "pulse": "Pulse"}
        if selected_trajectory != "custom" and st.session_state.dynamics_loaded:
            _customize_per_state = st.checkbox(
                "Set the reference per state", value=False, key="configure_customize_per_state",
                help="Choose each state's reference type independently instead of applying one type to the "
                     "whole system. Amplitude/frequency/pulse timing below are shared by every state that "
                     "uses them. Left off, the Setup Agent's detected derivative pairs already keep a "
                     "position/velocity pair physically consistent (position ~ sin implies velocity ~ cos).",
            )
            if _customize_per_state:
                _ps_state_names = st.session_state.dynamics_summary.get("state_names", [])
                if not _ps_state_names:
                    _ps_state_names = [f"x{i}" for i in range(st.session_state.dynamics_summary.get("n_states", 0))]
                _ps_default_label = _per_state_labels.get(selected_trajectory, "Regulation")
                _ps_rows = pd.DataFrame({
                    "State": _ps_state_names,
                    "Reference": [_ps_default_label] * len(_ps_state_names),
                })
                _ps_edited = st.data_editor(
                    _ps_rows, hide_index=True, width="stretch", key="configure_per_state_traj_editor",
                    column_config={
                        "State": st.column_config.TextColumn(disabled=True),
                        "Reference": st.column_config.SelectboxColumn(options=list(_per_state_labels.values())),
                    },
                )
                _label_to_code = {v: k for k, v in _per_state_labels.items()}
                per_state_trajectory_modes = [_label_to_code[v] for v in _ps_edited["Reference"]]
                st.caption(
                    "Cosinusoidal is exactly d/dt[Sinusoidal] -- pair a position state's Sinusoidal with its "
                    "velocity state's Cosinusoidal to keep the reference physically consistent."
                )

        # Overshoot is computed per state, and only for states whose own
        # reference is constant (see agents/metrics.py) -- so it only goes
        # N/A for the whole run when EVERY state is tracking something that
        # moves, which is exactly what this checks.
        _all_states_moving = (
            all(m in ("sin", "cos", "pulse") for m in per_state_trajectory_modes)
            if per_state_trajectory_modes else selected_trajectory in ("sin", "pulse")
        )
        if _all_states_moving:
            st.caption(
                "ℹ️ With every state tracking a moving target, **Overshoot will show N/A for every "
                "iteration** -- that's expected, not a bug: overshoot is only defined relative to a FIXED "
                "target to swing past, and there isn't one here. Watch **IAE/ISE** (Convergence tab) instead "
                "for tracking-quality metrics that make sense for a moving reference."
            )

        traj_amplitude, traj_frequency = 0.5, 0.5
        traj_pulse_start, traj_pulse_end = 0.2, 0.7
        # The shared amplitude/frequency/pulse controls follow whatever is
        # actually in use -- the single trajectory type, OR any per-state
        # override that needs them.
        show_sin_controls = selected_trajectory == "sin" or bool(
            per_state_trajectory_modes and any(m in ("sin", "cos") for m in per_state_trajectory_modes)
        )
        show_pulse_controls = selected_trajectory == "pulse" or bool(
            per_state_trajectory_modes and "pulse" in per_state_trajectory_modes
        )
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

        if _launch_step == 1:
            st.divider()
            if st.button("Next: Guidance & Constraints →", type="primary", width="stretch",
                          key="launch_step1_next"):
                st.session_state.launch_step = 2
                st.rerun()

    if _launch_step >= 2:
        with st.expander(
            "2  ·  Guidance & Constraints" + ("   ✓" if _launch_step > 2 else ""),
            expanded=_launch_step == 2,
        ):
            # Optimization Focus decides what "best" MEANS for the whole run,
            # so it sits in the open rather than behind a collapsed
            # "(optional)" expander where it was easy to never notice.
            st.markdown('<div style="color:#f1f3f7; font-weight:700;">Guidance for the Agent</div>',
                        unsafe_allow_html=True)
            st.caption("Optional -- the defaults are sensible -- but always on screen, because these change "
                       "what the agents actually optimize for.")
            optimization_focus = st.selectbox(
                "Optimization Focus",
                options=list(OPTIMIZATION_FOCUS_LABELS.keys()),
                format_func=lambda k: OPTIMIZATION_FOCUS_LABELS[k],
                help="Determines what 'best result' means (both for the agents' own best-so-far "
                     "tracking and the Best Result tab) -- 'Balanced' considers MSE, overshoot, "
                     "settling time, and control effort together.",
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
            # Read the summary here rather than relying on a name bound further up
            # the page: this tab used to sit after the "1 - System" tab, which
            # happened to leave `summary` in module scope. That tab is gone.
            _cn_summary = st.session_state.dynamics_summary if st.session_state.dynamics_loaded else {}
            _cn_input_names = _cn_summary.get("input_names", [])
            _cn_state_names = _cn_summary.get("state_names", [])
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

        if _launch_step == 2:
            _bcol, _ncol = st.columns([1, 3])
            with _bcol:
                if st.button("← Back", key="launch_step2_back", width="stretch"):
                    st.session_state.launch_step = 1
                    st.rerun()
            with _ncol:
                if st.button("Next: Scenario & Tuning →", type="primary", width="stretch",
                              key="launch_step2_next"):
                    st.session_state.launch_step = 3
                    st.rerun()

    if _launch_step >= 3:
        with st.expander("3  ·  Scenario & Tuning", expanded=True):
            scenario_level = render_card_selector(
                options=[
                    {"value": 1, "icon": "\u25cf", "title": "Level 1 \u00b7 Nominal",
                     "desc": "Clean run, no noise, nominal plant."},
                    {"value": 2, "icon": "\u25d0", "title": "Level 2 \u00b7 Noise",
                     "desc": "Adds measurement noise to selected states."},
                    {"value": 3, "icon": "\u25c9", "title": "Level 3 \u00b7 Robust",
                     "desc": "Noise + physical parameter mismatch."},
                ],
                key="scenario_level", default_value=1,
            )

            _scn_state_names = st.session_state.dynamics_summary.get("state_names", []) if st.session_state.dynamics_loaded else []
            scenario_noise_std_value = None
            scenario_noise_state_mask = None
            scenario_robust_noise_fraction = None
            scenario_perturb_physical_params = True
            scenario_max_param_uncertainty = None

            if scenario_level == 2 and _scn_state_names:
                with st.expander("Noise settings (Level 2)", expanded=True):
                    st.caption(
                        "By default, every state gets the same modest additive Gaussian measurement noise "
                        "(~1% of that state's declared range each step). Edit which states are affected and "
                        "how much below."
                    )
                    _default_noise = float(st.session_state.get("_suggested_noise_std", 0.01))
                    scenario_noise_std_value = st.number_input(
                        "Noise standard deviation (applied to every selected state)",
                        min_value=0.0, value=_default_noise,
                        step=_default_noise / 10 if _default_noise > 0 else 0.001, format="%.5f",
                    )
                    _noisy_states = st.multiselect(
                        "States that receive noise", options=_scn_state_names, default=_scn_state_names,
                        help="States left unchecked stay noise-free (exact measurement) even at this scenario level.",
                    )
                    scenario_noise_state_mask = np.array([s in _noisy_states for s in _scn_state_names])

            elif scenario_level == 3 and _scn_state_names:
                with st.expander("Robustness settings (Level 3)", expanded=True):
                    st.caption(
                        "Level 3 tunes against a plant whose PHYSICAL PARAMETERS aren't exactly what the "
                        "model assumes -- the real robustness test -- plus measurement noise. The initial "
                        "state is the same as every other level: what makes this level harder is the "
                        "plant-model mismatch, not a different starting point."
                    )
                    _default_noise = float(st.session_state.get("_suggested_noise_std", 0.01))
                    scenario_noise_std_value = st.number_input(
                        "Noise standard deviation",
                        min_value=0.0, value=_default_noise,
                        step=_default_noise / 10 if _default_noise > 0 else 0.001, format="%.5f",
                        help="Level 3 applies half of this magnitude (see apply_scenario_level).",
                    )
                    _noisy_states = st.multiselect(
                        "States that receive noise", options=_scn_state_names, default=_scn_state_names,
                        key="robust_noisy_states",
                        help="States left unchecked stay noise-free even at this scenario level.",
                    )
                    scenario_noise_state_mask = np.array([s in _noisy_states for s in _scn_state_names])

                    st.divider()
                    scenario_perturb_physical_params = st.checkbox(
                        "Perturb physical parameters (plant-model mismatch)", value=True,
                        help="Simulates the tuned controller facing a REAL system whose physical parameters "
                             "(mass, length, damping, ...) differ from the model's. This is what makes Level 3 "
                             "Robust rather than a noisier Level 1.",
                    )
                    if scenario_perturb_physical_params:
                        _unc_pct = st.slider(
                            "Maximum parametric uncertainty (%)", 1, 100, 20, step=1,
                            help="Upper bound of each selected parameter's OWN random boost -- every "
                                 "parameter draws independently in [0%, this], so they don't all move by "
                                 "the same amount. The exact draw is re-rolled per run and reported as "
                                 "substituted formulas once the run starts.",
                        )
                        scenario_max_param_uncertainty = _unc_pct / 100.0
                        _perturb_keys = _perturbable_param_names(st.session_state.get("dyn"))
                        if _perturb_keys:
                            st.latex(_param_uncertainty_formula(
                                st.session_state.dyn, _perturb_keys, scenario_max_param_uncertainty))
                            st.caption(
                                f"Every other numeric parameter, in sorted order -- "
                                f"{', '.join(_perturb_keys)} -- each boosted by its own draw. The realized "
                                f"values appear as substituted formulas above the results once the run starts."
                            )
                        else:
                            st.caption("No numeric physical parameters found on this plugin to perturb.")

            st.divider()
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

            st.divider()
            suggested_Q = st.session_state.get("suggested_Q")
            suggested_R = st.session_state.get("suggested_R")

            # No Level-3 Q boost any more: it existed to compensate for the
            # larger initial error the old "push the initial state toward its
            # bounds" behavior created, and that push is gone (see
            # scenario_presets.apply_scenario_level) -- so the boost had
            # nothing left to compensate for.
            q_for_default = list(suggested_Q) if suggested_Q else [1.0] * n_states_hint

            with st.expander("Initial Parameters", expanded=False):
                if suggested_Q and suggested_R:
                    st.caption("Pre-filled from the Initial Setup Analysis (Bryson's rule) and used automatically "
                               "as the Actor's starting point -- edit freely to override.")
                else:
                    st.caption("No automatic suggestion available for this file -- using flat defaults as the "
                               "starting point. Edit freely.")
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

        if st.button("← Back to Guidance & Constraints", key="launch_step3_back"):
            st.session_state.launch_step = 2
            st.rerun()

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
            st.session_state.ui_snapshot_html = None
            st.session_state.ui_snapshot_name = None
            st.session_state.export_script_text = None
            st.session_state.export_script_name = None
            st.session_state.animation_gif_bytes = None
            st.session_state.animation_gif_name = None
            st.session_state.animation_description = None
            st.session_state.animation_render_note = None
            st.session_state.manual_best_iteration = None
            st.session_state.diagnostics_report = None
            st.session_state.diagnostics_chat_history = []

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
                robust_noise_fraction=scenario_robust_noise_fraction,
                perturb_physical_params=scenario_perturb_physical_params,
                max_param_uncertainty=scenario_max_param_uncertainty,
            )
            st.session_state.run_perturbed_params = run_perturbed_params
            if use_custom_initial_state and custom_initial_state is not None:
                dyn.config.default_initial_state = custom_initial_state.copy()

            entry_node = "evaluator" if seed_params else "actor"
            # max_iterations has to reach the builder as well as initial_state: it
            # sizes LangGraph's recursion budget so the run ends on the iteration
            # cap instead of GraphRecursionError (see graph/workflow.py).
            graph = build_ui_tuning_graph(dyn, cfg, entry_node=entry_node,
                                           max_iterations=max_iterations)

            focus_sentence = (
                "" if optimization_focus == "balanced"
                else f"Optimization focus: {OPTIMIZATION_FOCUS_LABELS[optimization_focus]}. "
                     f"Prioritize this above the other metrics when proposing parameters."
            )
            combined_guidance = "\n".join(s for s in (focus_sentence, user_guidance.strip()) if s)

            # default_model=LLM_MODEL is what makes the dollar cost non-zero: see
            # TokenUsageTracker.on_llm_end -- without a known model name to price
            # against, every call bucketed under "unknown" and cost stayed $0.00.
            st.session_state.token_tracker = TokenUsageTracker(default_model=LLM_MODEL) if TOKEN_TRACKING_AVAILABLE else None

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


def _snap_esc(value) -> str:
    return html_module.escape(str(value), quote=True)


def _snapshot_fig_data_uri(fig) -> str:
    """PNG data URI straight from the figure as it is already styled (dark).

    Unlike report_pdf.py's ``_figure_png`` -- which re-renders the same chart
    light for a printed page -- this keeps the on-screen look untouched. The
    whole point of a UI snapshot is that it looks like the dashboard, not
    like the PDF report.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Layout rules private to the snapshot document. These are additive, not a
# restyle: every color and spacing value below reads from the SAME custom
# properties (--bg, --text, --line, ...) that DASHBOARD_CSS defines in :root,
# so a palette change made to the live app carries into this export without
# either file needing to change to match the other.
_SNAPSHOT_EXTRA_CSS = """
  /* DASHBOARD_CSS sets the dark background on `body` but the foreground
     color only on `.stApp` / specific [data-testid] selectors -- real
     Streamlit elements this exported document doesn't have. Without an
     explicit color here, plain elements (the <table> below, the Np/Nc/Q/R
     line) fall back to the browser default of black-on-black. Component
     classes that DO exist in DASHBOARD_CSS (.metric-card .value, .r-text,
     .log-entry, ...) already set their own color and are unaffected. */
  body { padding: 0; color: var(--text); }
  .snap-page { max-width: 1080px; margin: 0 auto; padding: 28px 32px 64px; }
  .snap-title { font-size: 1.7rem; font-weight: 800; color: var(--text); margin: 0 0 4px 0; }
  .snap-meta { color: var(--text-3); font-size: 0.85rem; margin: 0 0 20px 0; }
  .snap-cards, .snap-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 0 0 22px 0; }
  .kv-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0; }
  .snap-fig { max-width: 100%; border-radius: 10px; border: 1px solid var(--line); margin: 6px 0 18px 0; display: block; }
  .snap-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; margin: 10px 0 20px 0; }
  .snap-table th, .snap-table td { border: 1px solid var(--line); padding: 6px 9px; text-align: left; }
  .snap-table thead th { background: var(--surface-2); color: var(--text-2);
      text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.72rem; }
  .snap-table tbody tr.row-bad  { background: rgba(242,97,122,0.12); }
  .snap-table tbody tr.row-warn { background: rgba(232,177,61,0.12); }
  .snap-note { color: var(--text-3); font-style: italic; }
  @media (max-width: 900px) { .snap-cards, .snap-grid, .kv-row { grid-template-columns: repeat(2, 1fr); } }
"""


def build_ui_snapshot_html() -> str:
    """Self-contained HTML clone of the dashboard as it looks with results on
    screen: the same summary cards, best-result panel, convergence and
    simulation charts, iteration history, agent reasoning, run log and token
    usage -- in the exact same dark theme, since this reuses ``DASHBOARD_CSS``
    (the identical stylesheet the live page injects) rather than a
    hand-approximation of it.

    This intentionally does NOT go through report_pdf.py: that module's
    row-building helpers feed labcd_pdfmaker's print-oriented ReportBuilder
    (light page, Times New Roman, page breaks) and exist to produce a
    genuinely different document -- the shareable written report. This
    function's only job is to freeze what is already rendered on screen, as
    one portable file with every chart embedded as a PNG data: URI, so it
    opens correctly with no server and no network, on any machine, later.
    """
    summary = st.session_state.get("dynamics_summary", {}) or {}
    results = st.session_state.get("results_data", []) or []
    best = _best_row()
    df = pd.DataFrame(results)

    # ---- summary cards row (mirrors render_summary_cards) ----
    ok_df = df[df["ok"]] if not df.empty else df
    n_ok = len(ok_df) if not df.empty else 0
    n_fail = len(df) - n_ok if not df.empty else 0
    n_unstable = int(df["unstable"].sum()) if not df.empty and "unstable" in df.columns else 0
    status = "Done" if results else "Ready"
    elapsed = "--"
    if st.session_state.get("run_started_at"):
        end = st.session_state.get("run_finished_at") or datetime.now()
        elapsed = f"{(end - st.session_state.run_started_at).total_seconds():.0f}s"

    cards = [
        ("System", summary.get("dynamics_class", "--"), ""),
        ("Status", status, "status-ready"),
        ("Best MSE (stable)", fmt_num(best["mse"]) if best else "--", "value-cyan"),
        ("Iterations", f"{n_ok} ok / {n_unstable} unstable / {n_fail} failed", "value-purple"),
        ("Elapsed", elapsed, "value-yellow"),
    ]
    cards_html = "".join(
        f'<div class="metric-card"><div class="label">{_snap_esc(label)}</div>'
        f'<div class="value {cls}">{_snap_esc(value)}</div></div>'
        for label, value, cls in cards
    )

    # ---- best result (mirrors render_best_result) ----
    best_html = '<p class="snap-note">No successful iteration yet.</p>'
    if best is not None:
        overshoot_txt = fmt_num(best["overshoot"]) if best.get("overshoot_meaningful", True) else "N/A"
        settling_txt = f"{fmt_num(best['settling'])}s" if best["settling"] != float("inf") else "N/A"
        best_html = (
            f'<div class="subheader">Best Result — Iteration {best["iteration"]}</div>'
            '<div class="kv-row">'
            f'<div class="metric-card"><div class="label">MSE</div><div class="value">{_snap_esc(fmt_num(best["mse"]))}</div></div>'
            f'<div class="metric-card"><div class="label">Overshoot</div><div class="value">{_snap_esc(overshoot_txt)}</div></div>'
            f'<div class="metric-card"><div class="label">Settling Time</div><div class="value">{_snap_esc(settling_txt)}</div></div>'
            f'<div class="metric-card"><div class="label">Stable</div><div class="value">{"Yes" if best.get("is_stable") else "No"}</div></div>'
            "</div>"
            f'<p><strong>Np:</strong> {best["np"]} &nbsp; <strong>Nc:</strong> {best["nc"]} &nbsp; '
            f'<strong>Q:</strong> {_snap_esc(best["Q_formatted"])} &nbsp; '
            f'<strong>R:</strong> {_snap_esc(best["R_formatted"])} &nbsp; '
            f'<strong>P:</strong> {_snap_esc(best["P_formatted"])} &nbsp; '
            f'<strong>Oscillations:</strong> {best["oscillation_count"]}</p>'
        )
        per_state_mse = best.get("per_state_mse") or {}
        if per_state_mse:
            per_state_overshoot = best.get("per_state_overshoot") or {}
            rows = "".join(
                f"<tr><td>{_snap_esc(k)}</td><td>{_snap_esc(fmt_num(v))}</td>"
                f'<td>{_snap_esc(fmt_num(per_state_overshoot.get(k)) if per_state_overshoot.get(k) is not None else "N/A")}</td></tr>'
                for k, v in per_state_mse.items()
            )
            best_html += (
                '<table class="snap-table"><thead><tr><th>State</th><th>MSE</th><th>Overshoot</th></tr></thead>'
                f"<tbody>{rows}</tbody></table>"
            )
        if best.get("simulation_data") is not None:
            fig = plot_simulation_results(
                best["simulation_data"], best["iteration"],
                summary.get("state_names", []), summary.get("input_names", []),
                u_bounds=st.session_state.get("run_u_bounds"), x_bounds=st.session_state.get("run_x_bounds"))
            if fig:
                best_html += f'<img class="snap-fig" src="{_snapshot_fig_data_uri(fig)}" alt="Best-iteration simulation">'
                plt.close(fig)

    # ---- convergence chart (mirrors the Convergence tab) ----
    conv_html = '<p class="snap-note">No successful iterations to chart.</p>'
    if not df.empty:
        fig = plot_convergence(df)
        if fig:
            conv_html = f'<img class="snap-fig" src="{_snapshot_fig_data_uri(fig)}" alt="Convergence">'
            plt.close(fig)

    # ---- iteration history (mirrors render_data_live's table) ----
    history_html = '<p class="snap-note">No iterations recorded.</p>'
    if results:
        rows = []
        for r in results:
            status_tag = "UNSTABLE" if r.get("unstable") else ("OK" if r["ok"] else "FAILED")
            row_cls = "row-bad" if r.get("unstable") else ("row-warn" if not r["ok"] else "")
            mse = fmt_num(r["mse"]) if r["ok"] else "--"
            overshoot = (fmt_num(r["overshoot"]) if r["ok"] and r.get("overshoot_meaningful", True)
                        else ("N/A" if r["ok"] else "--"))
            settling = f"{fmt_num(r['settling'])}s" if r["ok"] and r["settling"] != float("inf") else "N/A"
            stable = "Yes" if r["ok"] and r.get("is_stable") else "No"
            dt_txt = fmt_num(r["dt_mpc"]) if r.get("dt_mpc") is not None else "--"
            rows.append(
                f'<tr class="{row_cls}"><td>{r.get("iteration","")}</td><td>{_snap_esc(status_tag)}</td>'
                f'<td>{r.get("np","")}</td><td>{r.get("nc","")}</td><td>{_snap_esc(mse)}</td>'
                f'<td>{_snap_esc(overshoot)}</td><td>{_snap_esc(settling)}</td><td>{stable}</td>'
                f'<td>{_snap_esc(dt_txt)}</td></tr>'
            )
        history_html = (
            '<table class="snap-table"><thead><tr><th>Iter</th><th>Status</th><th>Np</th><th>Nc</th>'
            "<th>MSE</th><th>Overshoot</th><th>Settling</th><th>Stable</th><th>dt (s)</th></tr></thead>"
            f'<tbody>{"".join(rows)}</tbody></table>'
        )

    # ---- agent reasoning (mirrors render_reasoning_panel -- real classes, not an approximation) ----
    reasoning_entries = st.session_state.get("reasoning_entries", [])
    reasoning_html = '<p class="snap-note">No agent activity recorded.</p>'
    if reasoning_entries:
        blocks = []
        for entry in reversed(reasoning_entries):
            text, time_str = entry["text"], entry["time"]
            node_label = text.split("]")[0].lstrip("[") if text.startswith("[") else "INFO"
            css_class = _reasoning_node_class(text)
            body = text.split("]", 1)[1].strip() if "]" in text else text
            blocks.append(
                '<div class="reasoning-entry"><div class="r-header">'
                f'<span class="r-node {css_class}">{_snap_esc(node_label)}</span>'
                f'<span class="r-time">{_snap_esc(time_str)}</span></div>'
                f'<div class="r-text">{_snap_esc(body)}</div></div>'
            )
        reasoning_html = f'<div class="reasoning-container">{"".join(blocks)}</div>'

    # ---- run log (mirrors render_log_panel -- real classes) ----
    logs = st.session_state.get("logs", [])
    log_html = '<p class="snap-note">No log entries recorded.</p>'
    if logs:
        entries = []
        for entry in logs:
            node, message, time_str = entry["node"], entry["message"], entry["time"]
            cls = ("log-metric" if "METRIC" in node else "log-error" if "ERROR" in node
                  else "log-success" if "DONE" in node else "")
            entries.append(
                f'<div class="log-entry"><span class="log-time">[{_snap_esc(time_str)}]</span> '
                f'<span class="log-node">{_snap_esc(node)}:</span> <span class="{cls}">{_snap_esc(message)}</span></div>'
            )
        log_html = f'<div class="log-container">{"".join(entries)}</div>'

    # ---- token usage / cost ----
    usage_section = ""
    tracker = st.session_state.get("token_tracker")
    if tracker is not None:
        usage = tracker.snapshot()
        if usage["call_count"] > 0:
            cost = usage.get("cost_usd")
            usage_cards = [
                ("Total Tokens", f"{usage['total_tokens']:,}"),
                ("Prompt Tokens", f"{usage['prompt_tokens']:,}"),
                ("Completion Tokens", f"{usage['completion_tokens']:,}"),
                ("LLM Calls", str(usage["call_count"])),
                ("Estimated Cost", "n/a" if cost is None else f"${cost:.6f}"),
            ]
            usage_grid = "".join(
                f'<div class="metric-card"><div class="label">{_snap_esc(l)}</div><div class="value">{_snap_esc(v)}</div></div>'
                for l, v in usage_cards
            )
            usage_section = f'<div class="subheader">Token Usage and Cost</div><div class="snap-grid">{usage_grid}</div>'

    system_name = summary.get("dynamics_class", "Unknown System")
    # A literal "·" character, not the "&middot;" entity: this string goes
    # through _snap_esc() below, which escapes "&" -- an entity written here
    # would come out as literal text "&middot;" instead of a rendered dot.
    meta = (f"{len(results)} iteration(s) · {n_ok} successful · "
            f"{'stopped by user' if st.session_state.get('stopped_by_user') else 'completed'}")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        f"<title>{_snap_esc('MPC Run - ' + system_name)}</title>"
        f"{DASHBOARD_CSS}<style>{_SNAPSHOT_EXTRA_CSS}</style>"
        '</head><body><div class="snap-page">'
        f'<div class="snap-title">{_snap_esc(system_name)} &middot; Tuning Run</div>'
        f'<div class="snap-meta">{_snap_esc(meta)} &middot; saved {_snap_esc(generated_at)}</div>'
        f'<div class="snap-cards">{cards_html}</div>'
        f"{best_html}"
        f'<div class="subheader">Convergence</div>{conv_html}'
        f'<div class="subheader">Iteration History</div>{history_html}'
        f'<div class="subheader">Agent Reasoning</div>{reasoning_html}'
        f'<div class="subheader">Run Log</div>{log_html}'
        f"{usage_section}"
        "</div></body></html>"
    )


def render_report_section():
    """The 'Generate Report', 'Export Script', 'Generate Animation', and
    'Save UI Snapshot' buttons -- all disabled while a run is active, enabled
    once it's stopped or finished (there's a best result to work from).
    Report: runs the Report Agent (an LLM call analyzing the actual results)
    and builds a PDF via reportlab -- a written document, its own layout.
    Export: builds a self-contained .py file combining the actual
    dynamics/MPC source with the tuned best parameters, runnable on the
    user's own machine. Animation: the Animation Agent (see
    agents/animation_agent.py) writes a small, heavily-sandboxed per-frame
    drawing function, and ordinary Python code renders it against the
    ACTUAL best-result trajectory into a GIF -- replaces an earlier static-
    SVG-schematic feature that wasn't useful enough to keep. UI Snapshot:
    no LLM call, no reportlab -- freezes the dashboard itself (see
    build_ui_snapshot_html), the same dark cards/panels/charts already on
    screen, as one portable .html file to send someone. All four outputs
    are cached in session_state so re-rendering this section (every script
    rerun) doesn't redo the work unless the user explicitly clicks the
    button again.
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

    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
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

    with col5:
        # No LLM call and no reportlab dependency -- this only re-packages
        # data and figures the run already produced, so it stays available
        # even when Generate Report is disabled (REPORT_FEATURE_AVAILABLE
        # false, or the LLM not configured for a fresh analysis pass).
        snapshot_help = "Available once the run stops or finishes."
        if has_results:
            snapshot_help = ("Saves the dashboard exactly as it looks right now -- summary cards, best "
                              "result, both charts, the full iteration table, agent reasoning and the run "
                              "log -- as one self-contained .html file. Same dark theme, opens in any "
                              "browser, nothing else needed.")
        if st.button("Save UI Snapshot", icon=":material/photo_camera:", use_container_width=True,
                      disabled=not has_results or st.session_state.running, help=snapshot_help):
            with st.spinner("Building snapshot..."):
                try:
                    st.session_state.ui_snapshot_html = build_ui_snapshot_html()
                    st.session_state.ui_snapshot_name = (
                        f"{st.session_state.dynamics_summary.get('dynamics_class', 'mpc')}_snapshot_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.html"
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Snapshot generation failed: {e}")

    if st.session_state.get("report_pdf_bytes"):
        with col1:
            st.download_button(
                "Download Report PDF", st.session_state.report_pdf_bytes,
                file_name=st.session_state.get("report_pdf_name", "mpc_report.pdf"),
                mime="application/pdf", use_container_width=True, key="report_pdf_download",
            )

    if st.session_state.get("ui_snapshot_html"):
        with col1:
            st.download_button(
                "Download UI Snapshot (HTML)", st.session_state.ui_snapshot_html,
                file_name=st.session_state.get("ui_snapshot_name", "mpc_snapshot.html"),
                mime="text/html", use_container_width=True, key="ui_snapshot_download",
            )


if st.session_state.running:
    _stop_col1, _stop_col2 = st.columns([5, 1])
    with _stop_col2:
        if st.button("Stop", icon=":material/stop:", use_container_width=True,
                      help="Stops after the current agent step finishes. Everything completed so far is kept -- "
                           "nothing is discarded -- and the last parameters become available to Manual Simulation."):
            st.session_state.stop_requested = True

render_report_section()


def _build_diagnostics_context() -> str:
    """Everything the Diagnostics Agent needs to give a SPECIFIC, grounded
    diagnosis instead of generic troubleshooting: the system and constraints
    the user actually declared, the full per-iteration history (params,
    metrics, per-state MSE, instability reason, solver health), AND every
    other agent's own reasoning text -- so e.g. "why did Q3 increase" can be
    answered by quoting the Actor's own stated reason for that change,
    instead of guessing from the numbers alone.

    Deliberately NOT gated behind scan_for_issues() finding anything --
    this is passed to the LLM on every report/chat call regardless, since a
    perfectly clean-looking run can still raise questions ("why this
    strategy switch"), and a genuinely bad run (infeasible QP, divergence,
    hit the iteration cap without converging) needs this same data to
    explain *why*, not just *that*.

    Kept as plain text (not a nested structure) -- one string slotted
    straight into the prompt template. Not aggressively truncated: the
    tuning loop is capped at 30 iterations by the UI's own slider, so this
    never grows unbounded, and the user has said LLM cost is not a concern
    here.
    """
    summary = st.session_state.get("dynamics_summary") or {}
    dyn = st.session_state.get("dyn")
    state_names = summary.get("state_names") or []
    input_names = summary.get("input_names") or []

    def _bounds_text(names: list, bounds) -> str:
        if not bounds or not names:
            return "(none declared -- unconstrained)"
        lo, hi = bounds
        return ", ".join(
            f"{n}: [{lo[i]:.4g}, {hi[i]:.4g}]" for i, n in enumerate(names) if i < len(lo) and i < len(hi)
        )

    lines: List[str] = []
    lines.append(f"System: {summary.get('dynamics_class', '?')}")
    lines.append(f"States: {state_names}")
    lines.append(f"Inputs: {input_names}")
    lines.append(f"Declared state bounds: {_bounds_text(state_names, st.session_state.get('run_x_bounds'))}")
    lines.append(f"Declared input bounds: {_bounds_text(input_names, st.session_state.get('run_u_bounds'))}")

    scenario_level = st.session_state.get("run_scenario_level")
    if scenario_level:
        lines.append(f"Scenario: Level {scenario_level} ({SCENARIO_LEVEL_NAMES.get(scenario_level, '?')})")
    lines.append(f"Trajectory mode: {st.session_state.get('run_trajectory_mode', '?')}")
    if dyn is not None:
        lines.append(f"Initial state actually used this run: {[round(float(v), 4) for v in dyn.config.default_initial_state]}")
        lines.append(f"Target state: {[round(float(v), 4) for v in dyn.config.default_target]}")
    lines.append(f"dt_mpc (starting): {st.session_state.get('suggested_dt')}")

    lines.append(f"\nRun status: {'still running' if st.session_state.running else 'stopped/finished'}")
    stop_reason = st.session_state.get("run_stop_reason")
    if stop_reason:
        lines.append(
            "Stop reason: " + {
                "user": "user pressed Stop",
                "max_iterations": "hit the Max Iterations cap without an earlier accept/stop decision",
                "terminator": "the Terminator/Juror decided to end the run",
                "recursion_limit": "hit the internal step budget (LangGraph recursion limit) before a stop condition",
            }.get(stop_reason, stop_reason)
        )

    rows = st.session_state.get("results_data") or []
    lines.append(f"\nPer-iteration history ({len(rows)} iteration(s), chronological):")
    for r in rows:
        if not r.get("ok"):
            lines.append(f"  Iter {r.get('iteration')}: FAILED -- {r.get('error')}")
            continue
        tag = "UNSTABLE" if r.get("unstable") else ("SETTLED" if r.get("success") else "running-out-clock")
        reason = f" ({r['unstable_reason']})" if r.get("unstable") and r.get("unstable_reason") else ""
        per_state = ", ".join(f"{k}={v:.4g}" for k, v in (r.get("per_state_mse") or {}).items())
        solver = r.get("solver_diagnostics") or {}
        solver_text = (
            f"  solver={solver.get('solved', 0)} clean/{solver.get('solved_inaccurate', 0)} "
            f"inaccurate/{solver.get('other', 0)} other" if solver else ""
        )
        lines.append(
            f"  Iter {r.get('iteration')} [{r.get('strategy', '?')}] {tag}{reason}: "
            f"Np={r.get('np')} Nc={r.get('nc')} Q={r.get('Q_formatted')} R={r.get('R_formatted')} "
            f"MSE={fmt_num(r.get('mse'))} (per-state: {per_state}) Overshoot={fmt_num(r.get('overshoot'))} "
            f"Settling={fmt_num(r.get('settling'))} Effort={fmt_num(r.get('effort'))}{solver_text}"
        )

    reasoning = st.session_state.get("reasoning_entries") or []
    if reasoning:
        lines.append(
            "\nAgent reasoning log (every Actor/Critic/Terminator/Juror/Scenarist decision this run, "
            "chronological -- each one's OWN stated reason for what it changed and why):"
        )
        for entry in reasoning:
            lines.append(f"  [{entry.get('time')}] {entry.get('text')}")

    return "\n".join(lines)


def render_diagnostics_tab():
    """Diagnostics Agent, now a fixed, always-reachable tab instead of a
    banner that only ever appeared (and only above the tab strip, easy to
    scroll past) when the deterministic scan found something. This tab is
    always in the tab strip so there is one consistent place to check --
    including when a run failed outright (run_error), which the banner
    version never surfaced at all beyond the raw traceback shown above the
    tabs.

    Three layers, same as before plus one new one:
      1. scan_for_issues() -- deterministic, free, runs on every render.
      2. generate_diagnostics_report() -- one LLM call, on demand, for
         grounded explanations/recommendations per detected category.
      3. NEW: a free-form chat (chat_about_issues) -- lets the user ask
         follow-up questions or describe what they've already tried,
         grounded in the same detected issues/report/raw error, instead of
         only ever reading the one-shot report.
    """
    st.markdown('<div class="subheader">Diagnostics</div>', unsafe_allow_html=True)
    st.caption("What went wrong (if anything) this run, the agent's recommendations, and a place to ask "
               "follow-up questions about it.")

    if not DIAGNOSTICS_FEATURE_AVAILABLE:
        st.warning(f"Diagnostics Agent unavailable: {DIAGNOSTICS_FEATURE_ERROR}")
        return

    run_error = st.session_state.get("run_error")
    if run_error:
        st.warning("This run failed with an error (see the full message above the tabs). "
                   "Ask below and the agent will look at it.")

    findings = scan_for_issues(
        st.session_state.get("logs", []), st.session_state.results_data, st.session_state.get("last_outputs", {}),
    ) if st.session_state.results_data else {}
    run_context = _build_diagnostics_context()

    if findings:
        categories_text = ", ".join(f"{v['count']}\u00d7 {k.replace('_', ' ')}" for k, v in findings.items())
        with st.container(border=True):
            st.warning(f"\u26a0\ufe0f **{len(findings)} issue type(s) detected this run** ({categories_text})")
            if st.button("Get detailed recommendations", key="diagnostics_get_recs"):
                with st.spinner("Diagnostics Agent analyzing..."):
                    st.session_state.diagnostics_report = generate_diagnostics_report(
                        findings, len(st.session_state.results_data), run_context=run_context,
                        tracker=st.session_state.get("token_tracker"))
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
    elif not run_error:
        st.success("No issues detected by the automatic scan so far.")

    st.divider()
    st.markdown("**Ask the Diagnostics Agent**")
    st.caption("Ask about a specific error, or describe what you've already tried -- the agent sees the "
               "detected issues, its own report above (if generated), and this run's raw error, if any.")

    for turn in st.session_state.diagnostics_chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    user_message = st.chat_input("e.g. \"why does the rate limit keep happening?\" or paste an error")
    if user_message:
        st.session_state.diagnostics_chat_history.append({"role": "user", "content": user_message})
        try:
            with st.spinner("Thinking..."):
                reply = chat_about_issues(
                    user_message, findings, st.session_state.get("diagnostics_report"),
                    st.session_state.diagnostics_chat_history[:-1],
                    run_error=run_error, run_context=run_context, tracker=st.session_state.get("token_tracker"),
                )
        except Exception as e:  # noqa: BLE001
            reply = f"Couldn't get a reply: {e}"
        st.session_state.diagnostics_chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


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
    # Manual Simulation is a real tab now, not an always-present one: it only
    # makes sense to poke at once a run has actually stopped or finished (its
    # fields are pre-filled from that run's last parameters), so it's only
    # ADDED to the tab strip once st.session_state.running is False --
    # before that, or mid-run, it simply isn't in the list at all. Diagnostics
    # is always present -- a fixed, predictable place to check for issues and
    # ask about them, regardless of whether the automatic scan found anything
    # THIS render.
    # Diagnostics and Manual Simulation are the two "go here when something
    # needs your attention" tabs, as opposed to the six read-only result
    # views before them -- so they're colored (Streamlit's own :red[...] /
    # :green[...] markdown, which tab labels render) and pushed to the right
    # edge by DASHBOARD_CSS's .stTabs rule, to read as a separate group
    # rather than six-plus-two of the same thing.
    tab_names = ["Live Run", "Convergence", "Simulation", "Best Result", "Agent Reasoning", "Data & Export",
                 ":red[Diagnostics]"]
    _manual_sim_available = not st.session_state.running
    if _manual_sim_available:
        tab_names.append(":green[Manual Simulation]")

    render_summary_cards()
    if st.session_state.get("run_perturbed_params"):
        # The realized plant-model mismatch for THIS run, as substituted
        # formulas rather than a prose list -- the Configure step could only
        # show the bound the boost is drawn from (the draw itself happens
        # once, seeded from the run's own random_seed, when the run starts).
        with st.container(border=True):
            st.markdown(
                '<div style="color:#f1f3f7; font-weight:700;">\U0001f527 Level 3 (Robust) '
                '&mdash; parametric uncertainty applied to this run</div>', unsafe_allow_html=True)
            st.caption("The controller is being tuned against a plant whose physical parameters differ from "
                       "the model's by exactly these amounts -- each drawn independently.")
            render_perturbed_params_formulas(st.session_state.run_perturbed_params)
    _tabs = st.tabs(tab_names)
    (tab_live, tab_convergence, tab_simulation, tab_best, tab_reasoning, tab_data, tab_diagnostics) = _tabs[:7]
    tab_manual_sim = _tabs[7] if _manual_sim_available else None

    current_iter = len(st.session_state.results_data)
    last_decision = ""
    if st.session_state.results_data:
        last = st.session_state.results_data[-1]
        last_decision = f"last: {'FAILED' if not last['ok'] else last['strategy'].upper()}"

    with tab_live:
        render_agent_flow_diagram(active_node=active_node, iteration=current_iter, last_decision=last_decision,
                                   reasoning_entries=st.session_state.get("reasoning_entries", []))
        if st.session_state.stopped_by_user:
            st.warning(
                "Run stopped by user after "
                f"{current_iter} iteration(s). Everything completed is kept below."
                + (" The last parameters have been carried over to the Manual Simulation section below."
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

    with tab_diagnostics:
        render_diagnostics_tab()

    # Manual Simulation used to live buried at the bottom of the Simulation
    # tab -- easy to miss exactly when it matters most (right after a run
    # stops or finishes, when its fields are freshly pre-filled with that
    # run's last parameters). It's a real tab of its own now, in the same
    # clickable strip as every other tab, right after Data & Export -- but
    # unlike the others it only EXISTS in that strip once the run has
    # actually stopped/finished (tab_manual_sim is None while running; see
    # _manual_sim_available above), so it becomes available exactly when
    # there's something worth using it for.
    if tab_manual_sim is not None:
        with tab_manual_sim:
            render_manual_simulation_tab()


def render_token_usage_summary():
    """Full LLM token usage breakdown for this run -- see llm_base.py's
    TokenUsageTracker for how this is collected (a LangChain callback
    attached to every agent's LLM call, Actor/Critic/Terminator/Juror plus
    the on-demand Report/Animation/Diagnostics agents).

    Cost comes from ``labcd_agents.CostCalculator`` -- the shared LabCD price
    table, the same one AgentPlant reports its spend from -- via
    TokenUsageTracker.snapshot(). Models the table does not recognise are
    listed as unpriced rather than silently counted as free, which is the
    failure mode that would otherwise understate a run's real cost with no
    indication anything was missing.

    The manual estimator below is kept for exactly that case: a model the
    shared table has not caught up with yet still gets a number, entered from
    whatever the provider's current pricing page says.
    """
    st.divider()
    st.markdown('<div class="subheader">LLM Token Usage</div>', unsafe_allow_html=True)
    tracker = st.session_state.get("token_tracker")
    if tracker is None or tracker.snapshot()["call_count"] == 0:
        st.caption("No LLM calls recorded yet for this run.")
        return

    usage = tracker.snapshot()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Tokens", f"{usage['total_tokens']:,}")
    with c2:
        st.metric("Prompt Tokens", f"{usage['prompt_tokens']:,}")
    with c3:
        st.metric("Completion Tokens", f"{usage['completion_tokens']:,}")
    with c4:
        st.metric("LLM Calls", usage["call_count"])
    with c5:
        _cost = usage.get("cost_usd")
        st.metric("Estimated Cost", "n/a" if _cost is None else f"${_cost:.6f}")

    _unpriced = usage.get("unpriced_models") or []
    if _cost is None:
        st.caption("Cost needs the shared `labcd_agents` package, which isn't installed here.")
    elif _unpriced:
        st.caption(
            "The cost above excludes " + ", ".join(f"`{m}`" for m in _unpriced) +
            " -- not in the shared price table. Use the manual estimator below for those."
        )
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

    with st.expander("Estimate cost manually", expanded=False):
        st.caption(
            "For a model the shared price table doesn't cover yet -- enter the current rate from your "
            "provider's own pricing page. This is independent of the figure above."
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
        st.session_state.run_stop_reason = "user"
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
            # The graph only reaches END through the Juror, and the Juror is
            # forced to accept_and_end once iteration >= max_iterations.
            st.session_state.run_stop_reason = (
                "max_iterations"
                if st.session_state.get("iteration", 0) >= st.session_state.get("run_max_iterations", 0)
                else "terminator"
            )
            st.session_state.graph_iterator = None
            if st.session_state.latest_params:
                prefill_manual_sim_from_params(st.session_state.latest_params)
            st.rerun(scope="app")  # run truly ends here -- same reasoning as above
        except GraphRecursionError as e:
            # The superstep budget, not a failure. Everything evaluated so far
            # is still valid, so this ends the run the same way the Stop button
            # does -- keeping best-so-far -- instead of taking the run_error
            # path below, which discards a run that may be many good iterations
            # deep. graph/workflow.py sizes the budget from max_iterations so
            # this should now only fire on a genuine runaway loop.
            best = _best_row()
            detail = (f"best MSE so far: {fmt_num(best.get('mse'))} at iteration {best.get('iteration')}"
                      if best else "no successful iteration to keep")
            log.warning("Run hit the LangGraph recursion limit: %s", e)
            st.session_state.logs.append({
                "time": time.strftime("%H:%M:%S"), "node": "STOP",
                "message": f"Stopped: recursion_limit reached before a stop condition -- {detail}.",
            })
            st.session_state.run_stop_reason = "recursion_limit"
            st.session_state.running = False
            st.session_state.run_finished_at = datetime.now()
            st.session_state.graph_iterator = None
            if st.session_state.latest_params:
                prefill_manual_sim_from_params(st.session_state.latest_params)
            st.rerun(scope="app")
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
                            "unstable_reason": metrics.get("Unstable_Reason"),
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
