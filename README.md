# Agentic Controller Designer

LLM-driven plant modeling and control-design pipelines in one repo:

| Module | What it designs | Core package | UI entry |
|--------|-----------------|--------------|----------|
| **AgentPlant** | Natural-language plant → runnable `dynamics(t, x, u)` draft | `backend_core/AgentPlant/` | `frontend_streamlit/agent_plant_app.py` |
| **AgentAdaptive** | Adaptive nonlinear controllers (SMC / Backstepping + RBF + disturbance observer) | `backend_core/AgentAdaptive/` | `frontend_streamlit/streamlit_app.py` |
| **AgentMPC** | Generic, plugin-based Model Predictive Control with multi-agent auto-tuning | `backend_core/AgentMPC/` | `frontend_streamlit/agent_mpc_app.py` |

Shared foundation: `packages/labcd_agents` (provider-agnostic LLM clients, tokens, prompts).

---

## Repository layout

```
.
├── frontend_streamlit/
│   ├── agent_plant_app.py               # AgentPlant UI (plant → dynamics)
│   ├── run_agent_plant_ui.py            # launcher → agent_plant_app (port 8503)
│   ├── streamlit_app.py                 # AgentAdaptive UI
│   ├── agent_mpc_app.py                 # AgentMPC full UI
│   ├── agent_mpc_ui_components.py       # AgentMPC minimal-decision UI
│   ├── run_agent_mpc_ui.py              # launcher → agent_mpc_app (port 8501)
│   └── run_agent_mpc_ui_components.py   # launcher → ui_components (port 8502)
├── backend_core/
│   ├── AgentPlant/                      # plant-model agent (NL → dynamics.py)
│   │   ├── agent.py                     # PlantModelAgent
│   │   ├── promptTemplate.yaml          # system prompt
│   │   ├── run_cli.py                   # terminal chatbot
│   │   └── GUIDE.md                     # product / UX guide
│   ├── AgentAdaptive/                   # adaptive nonlinear control core
│   │   ├── agents/                      # Clarifier, Designer, Tuner, Report Writer
│   │   ├── controller/                  # SMC / Backstepping derivation + simulation
│   │   ├── tools/                       # system_spec, scoring, PDF, pricing
│   │   ├── content/                     # stability proofs
│   │   ├── prompts/                     # YAML system prompts
│   │   └── data/                        # example plants
│   └── AgentMPC/                        # MPC design + LLM tuning core
│       ├── agents/                      # Actor, Critic, Scenarist, Terminator, Juror, …
│       ├── dynamics/                    # BaseDynamics contract + plugin loader
│       ├── mpc/                         # Config, GenericMPC, Jacobian, OSQP solver
│       ├── graph/                       # LangGraph tuning workflow
│       ├── utils/                       # logging
│       ├── tests/                       # unit / regression tests
│       └── run_agents.py                # CLI full tuning loop
├── packages/
│   └── labcd_agents/                    # shared LLM factory (editable install)
├── requirements.txt
├── packages.txt                         # optional system packages (e.g. TeX Live)
└── .env.example                         # API keys / default model (copy to .env)
```

Imports are absolute from the repo root, e.g. `from backend_core.AgentMPC.mpc.config import Config`.
Always run with `PYTHONPATH=.` (or use the provided launchers).

---

## Setup

```bash
# 1) Shared LLM package (use [all] if you switch providers in the UI)
pip install -e "packages/labcd_agents[all]"

# 2) Project dependencies
pip install -r requirements.txt
```

Copy `.env.example` → `.env` and set at least one provider key (`OPENAI_API_KEY`, `GROQ_API_KEY`, …).

Optional:

- **AgentAdaptive PDF reports** — LaTeX toolchain (`xelatex`). See `packages.txt`.
- **AgentMPC analytic Jacobians** — install `torch` (otherwise finite differences are used).

---


## AgentPlant

Conversational plant-model agent: describe a physical system in natural language;
the agent clarifies only as needed, then drafts runnable `dynamics(t, x, u)` code
you can revise until you accept (or a draft limit is reached). Download `dynamics.py`
when status is `complete`.

Statuses: `continue` (one clarifying question) → `draft` (code + note) → `complete` (accepted).

### Run UI

```bash
PYTHONPATH=. streamlit run frontend_streamlit/agent_plant_app.py
# or:
python frontend_streamlit/run_agent_plant_ui.py
```

### CLI

```bash
PYTHONPATH=. python backend_core/AgentPlant/run_cli.py
PYTHONPATH=. python backend_core/AgentPlant/run_cli.py --model gpt-4o
```

Default model: env `LABCD_DEMO_MODEL` (fallback `gpt-4o-mini`). Keys from repo-root `.env` via `labcd_agents`.

See `backend_core/AgentPlant/GUIDE.md` for product rules and example dialogues.

---

## AgentAdaptive

LLM-driven design of adaptive nonlinear controllers (Sliding Mode Control and
Backstepping), with adaptive RBF-network uncertainty estimation and a
disturbance observer.

Paste your plant as JSON, fill in the sim-knobs form, and answer the
Clarifier’s questions about uncertainty/disturbance. The pipeline then
selects the method, derives and simulates the control law, and optionally
tunes gains against your weighted objectives.

### Pipeline

| # | Agent | Job |
|---|-------|-----|
| 0 | **Clarifier** | Reads plant JSON + form; asks about uncertainty/disturbance (up to 6 rounds). |
| 1 | **Designer** | Confirms the spec, picks SMC vs Backstepping, fills the extraction schema. Does not derive the law. |
| 2 | **Tuner** | Proposes new *tuning* parameters (never structural ones) from simulation metrics. |
| 3 | **Report Writer** | One call at the end: short plain-English abstract for the report. |

The control law is derived and simulated exactly **once** in Python, after the Designer’s extraction succeeds.

### Configuration

Agents are built via `backend_core/AgentAdaptive/agents/llm_factory.py` (wrapper around `labcd_agents.LLMFactory`).
Keys are read from the **repo-root** `.env` (see `.env.example`). A legacy fallback still looks for
`../plantAgent-master/.env` if present.

Optional per-role overrides (fall back to the Designer defaults):

- `OPENAI_MODEL`, `OPENAI_MODEL_CLARIFIER`, `OPENAI_MODEL_TUNER`, `OPENAI_MODEL_REPORTER`
- matching `OPENAI_MAX_TOKENS_*` ceilings
- optional per-role keys: `OPENAI_API_KEY_CLARIFIER`, `OPENAI_API_KEY_TUNER`, `OPENAI_API_KEY_REPORTER`

### Run

```bash
PYTHONPATH=. streamlit run frontend_streamlit/streamlit_app.py
```

CLI (optional):

```bash
PYTHONPATH=. python -m backend_core.AgentAdaptive.agents.cli
```

---

## AgentMPC

Plugin-based linearised MPC (OSQP) with an optional multi-agent LLM auto-tuning loop (Actor / Critic / Scenarist / Terminator / Juror and supporting agents).

- **Core only** (`dynamics/`, `mpc/`, `agents/evaluator.py`, `agents/numeric_tuner.py`): needs `numpy`, `scipy`, `osqp` — no LLM stack.
- **Full agents + UI**: also needs `langgraph`, `labcd_agents`, Streamlit extras (`reportlab`, `pillow`, `pandas`, …).

### Layout (core)

```
backend_core/AgentMPC/
├── dynamics/     # BaseDynamics, DynamicLoader, trajectory loader, example plugin
├── mpc/          # Config, GenericMPC, Jacobian, QPSolver (OSQP)
├── agents/       # LLM nodes + numeric evaluator / tuner / reports
├── graph/        # LangGraph state + workflow
├── utils/        # logging
└── tests/
```

### Run UI

```bash
PYTHONPATH=. streamlit run frontend_streamlit/agent_mpc_app.py
# or one-command launcher (installs deps, opens browser):
python frontend_streamlit/run_agent_mpc_ui.py

# Minimal-decision UI (port 8502):
python frontend_streamlit/run_agent_mpc_ui_components.py
```

### CLI tuning loop

```bash
PYTHONPATH=. python backend_core/AgentMPC/run_agents.py
```

### Tests

```bash
PYTHONPATH=. pytest backend_core/AgentMPC/tests -v
```

### Quick start (numeric only, no LLM)

```python
from pathlib import Path
from backend_core.AgentMPC.dynamics.loader import DynamicLoader
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentMPC.agents.numeric_tuner import make_objective, random_search, coordinate_refine

plugin_path = Path("backend_core/AgentMPC/dynamics/plugins/example_pendulum.py")
plugin = DynamicLoader.load_from_path(str(plugin_path.resolve()))
dyn = plugin.create_dynamics()

cfg = Config()
cfg.mpc.prediction_horizon = 12
cfg.data.dt_mpc = 0.02
cfg.data.simulation_time = 3.0

objective = make_objective(dyn, cfg)
result = random_search(objective, n_states=dyn.n_states, n_inputs=dyn.n_inputs, n_trials=40)
refined = coordinate_refine(objective, result.best_params, n_iters=30)
```

Dynamics plugins implement `create_config()` and a `BaseDynamics` subclass. See
`backend_core/AgentMPC/dynamics/plugins/example_pendulum.py`. Symbols
(`BaseDynamics`, `SystemConfig`, `np`) are injected by `DynamicLoader` when the
file is loaded as a plugin.

---

## Dependencies at a glance

| Concern | Packages |
|---------|----------|
| Shared LLM | `labcd_agents` (editable from `packages/`), `langchain*`, `pydantic`, `pyyaml` |
| AgentAdaptive core | `sympy`, `numpy`, `matplotlib` |
| AgentAdaptive PDF | system TeX (`packages.txt`) + existing Python stack |
| AgentMPC core | `numpy`, `scipy`, `osqp` |
| AgentMPC agents / graph | `langgraph`, `labcd_agents` |
| AgentMPC UI | `streamlit`, `reportlab`, `pillow`, `pandas`, `matplotlib` |
| Optional | `torch` (AgentMPC analytic Jacobians), `pytest` |

All of the above (except optional `torch` and system TeX) are listed in `requirements.txt`.
