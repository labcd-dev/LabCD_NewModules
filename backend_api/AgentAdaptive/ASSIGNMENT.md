# Assignment: FastAPI Layer for AgentAdaptive

## Purpose

Expose AgentAdaptive as a first-class service under `backend_api` in this repository, with contracts and organisation that align with LabCD_Application’s platform FastAPI layer and with the sibling AgentPlant implementation already present in NewModules.

The goal is to make adaptive / nonlinear control design (SMC / Backstepping + RBF + disturbance observer) available to the React frontend, external clients, and orchestration flows without depending on the Streamlit UI.

There is **no pre-existing Adaptive FastAPI package** in LabCD_Application comparable to PlantModelChat. This assignment is therefore a **greenfield API** that must:

1. Follow Application conventions for long-running design jobs (job id, status, cancel, artefacts).
2. Follow the **local package layout** established by AgentPlant under `backend_api/AgentPlant/` (schemas, service adapter, store, router, standalone app).
3. Drive only the core in `backend_core/AgentAdaptive/`.

---

## Context

AgentAdaptive is a multi-stage pipeline:

| Stage | Agent | Role |
|-------|--------|------|
| 0 | Clarifier | Up to several rounds of questions about uncertainty / disturbance |
| 1 | Designer | Confirms spec, selects SMC vs Backstepping, fills extraction schema |
| 2 | (Python) | Control law derived and simulated once after design extraction |
| 3 | Tuner | Proposes tuning parameters from simulation metrics (optional loops) |
| 4 | Report Writer | Final plain-English abstract for the report |

Today the module is used mainly via Streamlit (`frontend_streamlit/agent_adaptive_app.py` / related entry). Core code lives in `backend_core/AgentAdaptive/`.

Reference structures:

- Platform patterns: https://github.com/labcd-dev/LabCD_Application/tree/master/backend_api  
  especially `http/routers/jobs.py`, `http/services/job_store.py`, and module routers such as `mulo.py` / `trimmer.py` / `silo.py`
- Local FastAPI template (chat-style, already implemented): `backend_api/AgentPlant/`
- Local core: `backend_core/AgentAdaptive/`

---

## Expected Outcomes

1. **Service entry point**  
   AgentAdaptive is reachable under a clear route family (e.g. `/api/adaptive/...`). Callers can submit a design run, answer clarification turns if required, track progress through design / simulation / tuning / report, and retrieve results without Streamlit.

2. **Job-oriented lifecycle (with optional interactive clarification)**  
   Design runs are modelled as jobs (compatible in spirit with Application’s `job_store` + `/jobs/{id}`). Status and intermediate progress are queryable. Clarification may be synchronous request/response turns *or* job state that waits for user input—document the chosen pattern and keep it consistent for the frontend. Final artefacts (design summary, metrics, figures, PDF/report payload) are retrievable by job id.

3. **Core driven from `backend_core/AgentAdaptive`**  
   HTTP adapters call Clarifier, Designer, Tuner, Report Writer, and controller/simulation code from the existing core. Do not reimplement control derivation or prompts inside the API package.

4. **Consistent integration**  
   Reuse (or stub with a clear merge path) platform ideas for authentication, project context, model allow-lists, and result storage. Do not invent divergent auth or job semantics that would block merge into LabCD_Application.

5. **Clear separation of concerns**  
   Thin HTTP layer: schemas, orchestration, job/conversation persistence. All adaptive-design logic stays in `backend_core`.

6. **Standalone runnable app**  
   Like AgentPlant, provide a standalone FastAPI app under this package so NewModules can run Adaptive without the full Application monolith. Prefer in-memory or local job storage for demos; structure so Application’s `job_store` / DB can replace it later.

7. **Ready for the product frontend**  
   Contracts are sufficient for a React client to submit plant JSON + sim knobs, handle clarification, show progress, and download/review reports—peer UX to other LabCD design modules.

---

## Suggested package layout (match AgentPlant)

Place the implementation under `backend_api/AgentAdaptive/`:

```
backend_api/AgentAdaptive/
  ASSIGNMENT.md          # this file
  README.md              # how to run, endpoint summary
  __init__.py
  schemas.py             # request/response / job status models
  service.py             # thin adapter → backend_core.AgentAdaptive
  job_store.py           # in-memory (or file) job + artefact store for standalone
  router.py              # FastAPI routes
  app.py                 # standalone uvicorn entry
  tests/                 # unit tests with mocked LLM / fixed plant fixtures
```

Route ideas (adjust names as needed, keep them stable once chosen):

| Method | Path (under API prefix) | Purpose |
|--------|-------------------------|---------|
| `POST` | `/adaptive/jobs` | Start a design job (plant JSON + form / knobs) |
| `GET` | `/adaptive/jobs/{job_id}` | Status + stage + progress |
| `POST` | `/adaptive/jobs/{job_id}/clarify` | Submit clarification answer (if interactive) |
| `POST` | `/adaptive/jobs/{job_id}/cancel` | Request cancel |
| `GET` | `/adaptive/jobs/{job_id}/results` | Final artefacts (metrics, plots metadata, report) |
| `GET` | `/health` | Liveness (on the standalone app) |

If a shared NewModules job router is introduced later, Adaptive should register as module `"adaptive"` (or similar) rather than forking a second global job system.

---

## Scope Boundaries

- **In scope:** FastAPI package under `backend_api/AgentAdaptive`; wiring to `backend_core/AgentAdaptive`; job + clarification contracts; standalone app; tests with mocks/fixtures; short README.
- **Out of scope:** Redesigning Clarifier/Designer/Tuner prompts or control math; building the React UI; replacing Streamlit in this assignment (Streamlit may keep working in parallel).
- Alignment with LabCD_Application job conventions and with `backend_api/AgentPlant/` layout is mandatory for a clean future merge.

---

## Source of truth

| Piece | Location |
|--------|----------|
| Adaptive core | `backend_core/AgentAdaptive/` (agents, controller, tools, prompts) |
| Streamlit reference UI | `frontend_streamlit/agent_adaptive_app.py` (and related) |
| Local API template | `backend_api/AgentPlant/` (schemas / service / store / router / app) |
| Application jobs pattern | `LabCD_Application`: `http/routers/jobs.py`, `http/services/job_store.py`, module services (`mulo_service`, `trimmer_service`, …) |
| Application platform root | https://github.com/labcd-dev/LabCD_Application/tree/master/backend_api |

---

## Success Criteria

- AgentAdaptive can be driven end-to-end over HTTP in this repository (submit → optional clarify → design/sim/tune → results).
- Core logic remains in `backend_core/AgentAdaptive`; the API package stays thin.
- Package layout and standalone run story are consistent with AgentPlant.
- Job/status/artefact shapes are close enough to Application’s job model that merge does not require a second client integration style.
- Tests cover the service adapter and store without requiring live LLM calls for the happy-path structure.

---

*This assignment defines strategic intent and required outcomes. Prefer the AgentPlant package shape and Application job conventions; implementation details may adapt to Adaptive’s multi-stage pipeline provided the outcomes above are met.*
