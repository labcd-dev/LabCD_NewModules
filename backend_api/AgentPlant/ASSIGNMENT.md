# Assignment: Finish the AgentPlant FastAPI Layer — Deliver the Unified Artifact

## Purpose

Complete the FastAPI layer under `backend_api/AgentPlant` so that a plant conversation can end in a **persisted, module-ready artifact** — the same hand-off that `frontend_streamlit/unified_app.py` already performs via Pre-Launch → `PlantCompiler` → `ArtifactStore`.

Today the API covers chat only (`/api/plant-model/chat` and conversation CRUD). The unified Streamlit workflow goes further: after AgentPlant emits `status=complete`, the user supplies pre-launch knobs, the plant is compiled, and an artifact is written that can bootstrap **AgentMPC** (dynamics plugin `.py`) and **AgentAdaptive** (`system_spec`). That final delivery step is still missing from the HTTP surface.

The goal is to expose the full Plant → Pre-Launch → Artifact path over HTTP so React clients, smoke scripts, and orchestration flows can obtain the same deliverable without Streamlit.

---

## Context

### What already exists

| Layer | Location | Role |
|-------|----------|------|
| Core agent | `backend_core/AgentPlant/` | `PlantModelAgent`, session state, drafts → complete |
| Compiler + artifact model | `backend_core/plant_compiler.py` | Validate plant, compile `Artifact` (MPC plugin source + adaptive spec + full payload) |
| Filesystem store | `backend_core/artifact_store.py` | `save` / `save_from_plant` / `load` / `list_artifacts` / `get_adaptive_spec` / plugin path |
| FastAPI (chat only) | `backend_api/AgentPlant/` | `app.py`, `router.py`, `schemas.py`, `service.py`, `conversation_store.py` |
| Tests | `backend_api/AgentPlant/tests/` | `test_api.py`, `test_service.py` (mocked LLM) |
| Smoke script | `scripts/smoke_plant_api.py` | Live chat smoke against uvicorn on `:8003` |
| Reference UX | `frontend_streamlit/unified_app.py`, `pre_launch_panel.py` | Plant stage → Pre-Launch form → artifact id → launch Adaptive / MPC |

### Unified workflow (source of truth for the hand-off)

1. **Plant** — multi-turn chat until `final_result` (`system_name`, `python_code`, preferably `metadata`).
2. **Pre-Launch** — module-agnostic sim knobs: `total_simulation_time`, `solver_sample_time`, `initial_state`, `default_target` (see `default_pre_launch` / `validate_pre_launch` in `plant_compiler`).
3. **Compile + persist** — `ArtifactStore.save_from_plant(plant_output, pre_launch)` → `artifact_id`, `{id}.json`, `{id}.py`.
4. **Downstream bootstrap**
   - **AgentMPC:** load plugin via `store.load_plugin_path(artifact_id)`.
   - **AgentAdaptive:** `store.get_adaptive_spec(artifact_id)` as starting `system_spec`.

The FastAPI layer must make steps 2–4 available after a completed plant conversation (or from an explicit plant payload), without requiring Streamlit session state.

---

## Expected Outcomes

1. **Artifact delivery endpoints**  
   After plant completion (or with an explicit plant + pre-launch body), clients can compile and persist an artifact and receive `artifact_id` plus metadata needed by MPC and Adaptive.

2. **Parity with unified Streamlit**  
   Same validation rules, same pre-launch shape, same `ArtifactStore` layout under a configurable base directory. Repair path: validation errors can be returned so a client can send them back into the plant chat (as the UI does with `plant_validation_errors`).

3. **Thin adapter over existing core**  
   Reuse `PlantCompiler`, `ArtifactStore`, and the existing chat service. Do not reimplement compilation or dynamics codegen inside the API package.

4. **Conversation linkage (optional but recommended)**  
   Allow compiling from a completed conversation id (read `final_result` from the in-memory store) and/or from a request body that carries plant + pre-launch. Keep standalone anonymous access consistent with current chat routes.

5. **List / load / bootstrap helpers**  
   At minimum: create artifact, get artifact JSON, list artifacts, resolve MPC plugin path (or return plugin source), resolve Adaptive `system_spec`. Align response shapes with what `unified_app` / `ArtifactStore` already produce.

6. **Tests and smoke coverage**  
   Extend `backend_api/AgentPlant/tests/` with mocked/fixture plant + pre-launch (no live LLM required for artifact routes). Optionally extend `scripts/smoke_plant_api.py` or add a focused artifact smoke once the server is up.

7. **Standalone app remains the entry point**  
   Keep `uvicorn backend_api.AgentPlant.app:app` as the way to run the service; new routes live under the existing API prefix (default `/api`).

---

## Suggested work (layout stays under AgentPlant)

Extend the existing package rather than creating a sibling service:

```
backend_api/AgentPlant/
  ASSIGNMENT.md          # this file
  app.py                 # already exists — no structural change required
  router.py              # add artifact / pre-launch routes
  schemas.py             # PreLaunch, ArtifactCreate, ArtifactSummary, etc.
  service.py             # chat adapter + thin compile/persist helpers
  conversation_store.py  # unchanged unless you attach artifact_id to complete convos
  tests/
    test_api.py          # extend
    test_service.py      # extend
    # optional fixtures for a minimal valid plant + metadata
```

### Route ideas (names can be adjusted; keep them stable once chosen)

Prefix remains the package router prefix (`/plant-model` under `LABCD_API_PREFIX`, default `/api`).

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/plant-model/artifacts` | Compile + persist from plant payload + pre-launch (and/or conversation_id) |
| `GET` | `/plant-model/artifacts` | List artifact summaries |
| `GET` | `/plant-model/artifacts/{artifact_id}` | Full artifact JSON |
| `GET` | `/plant-model/artifacts/{artifact_id}/plugin` | MPC plugin source or path metadata |
| `GET` | `/plant-model/artifacts/{artifact_id}/adaptive-spec` | AgentAdaptive `system_spec` |
| `POST` | `/plant-model/validate` | Optional: validate plant / pre-launch without persisting (supports repair loop) |

Wire artifact base directory via env (e.g. `LABCD_ARTIFACTS_DIR`) defaulting to the same `artifacts/` convention used by Streamlit.

**Request shape sketch for create:**

- `conversation_id` (optional) — if set and conversation is complete, use its `final_result` as plant.
- `plant` — `{ system_name, python_code, metadata? }` when not using conversation.
- `pre_launch` — `{ total_simulation_time, solver_sample_time, initial_state, default_target }`.

On validation failure, return structured errors (HTTP 422 or 400 with a clear body) suitable for feeding back into `/plant-model/chat`.

---

## Scope Boundaries

- **In scope:** Extending `backend_api/AgentPlant` so the artifact is delivered over HTTP; schemas; service helpers calling `PlantCompiler` / `ArtifactStore`; tests; short README notes if needed; optional smoke-script updates.
- **Out of scope:** Redesigning AgentPlant prompts or the compiler; implementing AgentMPC / AgentAdaptive FastAPI packages (see their own `ASSIGNMENT.md` files); building the React UI; replacing Streamlit (it may keep working in parallel).
- Do not move core logic out of `backend_core`. Do not invent a second artifact format that diverges from `ArtifactStore` / `unified_app`.

---

## Source of truth

| Piece | Location |
|--------|----------|
| Plant chat API (existing) | `backend_api/AgentPlant/` |
| Plant core | `backend_core/AgentPlant/` |
| Compiler + `Artifact` | `backend_core/plant_compiler.py` |
| Filesystem artifact store | `backend_core/artifact_store.py` |
| Unified hand-off UX | `frontend_streamlit/unified_app.py`, `frontend_streamlit/pre_launch_panel.py` |
| Existing tests / smoke | `backend_api/AgentPlant/tests/`, `scripts/smoke_plant_api.py` |
| Downstream consumers | `backend_api/AgentMPC/ASSIGNMENT.md`, `backend_api/AgentAdaptive/ASSIGNMENT.md` |

---

## Success Criteria

- A client can complete (or supply) a plant, submit pre-launch config, and receive a durable `artifact_id` from the AgentPlant API.
- The stored artifact can bootstrap AgentMPC (plugin) and AgentAdaptive (`system_spec`) the same way the unified Streamlit app does.
- Validation failures are explicit and usable for a repair turn back into plant chat.
- Core compilation and storage remain in `backend_core`; the API package stays a thin HTTP adapter.
- Tests cover create/list/get (and preferably adaptive-spec / plugin) without a live LLM.
- Existing chat routes and standalone `app.py` continue to work; smoke chat flow remains valid.
