# Assignment: FastAPI Layer for AgentMPC

## Purpose

Expose AgentMPC as a first-class service under `backend_api` in this repository, with contracts and organisation that align with LabCD_Application’s platform FastAPI layer and with the sibling AgentPlant implementation already present in NewModules.

The goal is to make MPC auto-tuning (multi-agent graph over a plugin dynamics + MPC solver) available to the React frontend, external clients, and orchestration flows without depending on the Streamlit UI.

There is **no pre-existing MPC FastAPI package** in LabCD_Application comparable to PlantModelChat. This assignment is therefore a **greenfield API** that must:

1. Follow Application conventions for long-running design jobs (job id, status, cancel, artefacts).
2. Follow the **local package layout** established by AgentPlant under `backend_api/AgentPlant/` (schemas, service adapter, store, router, standalone app).
3. Drive only the core in `backend_core/AgentMPC/`.

---

## Context

AgentMPC runs a multi-agent tuning workflow (Actor, Critic, Scenarist, Terminator, Juror, validators, report agents, …) on top of:

- `backend_core/AgentMPC/dynamics/` — plant plugin contract + loader  
- `backend_core/AgentMPC/mpc/` — config, controller, solver  
- `backend_core/AgentMPC/graph/` — LangGraph (or equivalent) workflow state  

Runs can be long-lived and produce metrics, trajectory data, reports, and export scripts. Today the module is used mainly via Streamlit (`frontend_streamlit/agent_mpc_app.py`, `agent_mpc_ui_components.py`, launchers).

Reference structures:

- Platform patterns: https://github.com/labcd-dev/LabCD_Application/tree/master/backend_api  
  especially `http/routers/jobs.py`, `http/services/job_store.py`, and module routers such as `mulo.py` / `trimmer.py` / `silo.py`
- Local FastAPI template (already implemented): `backend_api/AgentPlant/`
- Local core: `backend_core/AgentMPC/`
- Entry for offline runs: `backend_core/AgentMPC/run_agents.py`

---

## Expected Outcomes

1. **Service entry point**  
   AgentMPC is reachable under a clear route family (e.g. `/api/mpc/...`). Callers can submit a tuning job (dynamics reference or payload, MPC config, scenario/options), track progress through agent rounds, and retrieve results without Streamlit.

2. **Job-oriented lifecycle**  
   Tuning runs are asynchronous jobs. Status, intermediate progress (e.g. current agent/round, key metrics), cancel support, and final artefacts (metrics, reports, figures metadata, export payloads) are available through job endpoints. Associate jobs with a user/project id when provided; for standalone NewModules, an in-memory store is acceptable if the merge path to Application’s `job_store` is obvious.

3. **Core driven from `backend_core/AgentMPC`**  
   The API starts and observes the existing graph/workflow (`graph/workflow.py`, agents, dynamics loader, MPC solver). Do not reimplement tuning agents or the MPC numeric stack inside the API package.

4. **Consistent integration**  
   Align with Application job status vocabulary and artefact retrieval patterns where practical. Do not introduce a parallel global job system that would conflict on merge.

5. **Clear separation of concerns**  
   HTTP handling, schemas, and job orchestration stay thin. Dynamics plugins, MPC numerics, and LLM agents stay in `backend_core`.

6. **Standalone runnable app**  
   Provide a standalone FastAPI app under this package (same idea as `backend_api/AgentPlant/app.py`) so MPC can be served from NewModules alone.

7. **Ready for the product frontend**  
   Contracts support submit → monitor → review results, comparable to other LabCD design pipelines and usable by a React client.

---

## Suggested package layout (match AgentPlant)

Place the implementation under `backend_api/AgentMPC/`:

```
backend_api/AgentMPC/
  ASSIGNMENT.md          # this file
  README.md              # how to run, endpoint summary
  __init__.py
  schemas.py             # submit / status / results models
  service.py             # thin adapter → backend_core.AgentMPC (graph / run_agents)
  job_store.py           # in-memory (or file) job + artefact store for standalone
  router.py              # FastAPI routes
  app.py                 # standalone uvicorn entry
  tests/                 # unit tests with mocked agents / fixed dynamics fixtures
```

Route ideas (adjust names as needed, keep them stable once chosen):

| Method | Path (under API prefix) | Purpose |
|--------|-------------------------|---------|
| `POST` | `/mpc/jobs` | Start a tuning job |
| `GET` | `/mpc/jobs/{job_id}` | Status + progress metadata |
| `POST` | `/mpc/jobs/{job_id}/cancel` | Request cancel |
| `GET` | `/mpc/jobs/{job_id}/results` | Final artefacts (metrics, report, export) |
| `GET` | `/health` | Liveness (on the standalone app) |

Register conceptually as module `"mpc"` (or similar) if/when a shared job router is used.

**Dynamics input:** accept either a registered plugin id, an uploaded/pasted `dynamics` source, or a path convention already used by `backend_core/AgentMPC/dynamics/`. Document the chosen contract in the package README; prefer something the Streamlit UI already understands so parity is easy.

---

## Scope Boundaries

- **In scope:** FastAPI package under `backend_api/AgentMPC`; wiring to `backend_core/AgentMPC`; async job lifecycle; standalone app; tests with mocks/fixtures; short README.
- **Out of scope:** Redesigning the multi-agent graph, MPC solver, or dynamics plugin API; building the React UI; replacing Streamlit in this assignment.
- Alignment with LabCD_Application job conventions and with `backend_api/AgentPlant/` layout is mandatory for a clean future merge.

---

## Source of truth

| Piece | Location |
|--------|----------|
| MPC core | `backend_core/AgentMPC/` (agents, dynamics, mpc, graph, `run_agents.py`) |
| Streamlit reference UI | `frontend_streamlit/agent_mpc_app.py`, `agent_mpc_ui_components.py` |
| Local API template | `backend_api/AgentPlant/` (schemas / service / store / router / app) |
| Application jobs pattern | `LabCD_Application`: `http/routers/jobs.py`, `http/services/job_store.py`, module services |
| Application platform root | https://github.com/labcd-dev/LabCD_Application/tree/master/backend_api |

---

## Success Criteria

- AgentMPC can be driven end-to-end over HTTP in this repository (submit job → poll status → fetch results; cancel when supported).
- Core logic remains in `backend_core/AgentMPC`; the API package stays thin.
- Package layout and standalone run story are consistent with AgentPlant.
- Job/status/artefact shapes are close enough to Application’s job model that merge does not require a second client integration style.
- Tests cover the service adapter and store without requiring full multi-agent LLM runs for structural checks.

---

*This assignment defines strategic intent and required outcomes. Prefer the AgentPlant package shape and Application job conventions; implementation details may adapt to MPC’s multi-agent graph provided the outcomes above are met.*
