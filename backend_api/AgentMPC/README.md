# AgentMPC FastAPI

Job-oriented HTTP API for MPC auto-tuning (Scenarist / Actor / Critic / Juror
graph over a dynamics plugin + MPC solver).

Core logic lives in `backend_core/AgentMPC/`. This package is a thin adapter +
in-memory job store, matching the layout of `backend_api/AgentPlant/` and
`backend_api/AgentAdaptive/`.

## Run

```bash
# from repository root
PYTHONPATH=. uvicorn backend_api.AgentMPC.app:app --host 127.0.0.1 --port 8005
# or
PYTHONPATH=. python -m backend_api.AgentMPC.app
```

Health: `GET http://127.0.0.1:8005/health`

API prefix: `LABCD_API_PREFIX` (default `/api`).

Requires `GROQ_API_KEY` (or equivalent LLM config used by `backend_core.AgentMPC`)
for live agent runs. Tests mock the graph and do not need a key.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/mpc/jobs` | Start a tuning job |
| `GET` | `/api/mpc/jobs` | List jobs |
| `GET` | `/api/mpc/jobs/{job_id}` | Status + stage + progress |
| `POST` | `/api/mpc/jobs/{job_id}/cancel` | Request cancel |
| `GET` | `/api/mpc/jobs/{job_id}/results` | Final artefacts (params, MSE, history) |

### Job lifecycle

1. **Submit** with dynamics reference + options.
2. Graph runs in a background thread (`build_ui_tuning_graph` by default, or
   full `build_mpc_tuning_graph` when `options.use_ui_graph` is false).
3. Poll `GET .../jobs/{id}` for `status` / `stage` / `iteration` / `progress`.
4. When `completed` (or `failed` / `cancelled`), fetch `GET .../results`.

### Dynamics input

Provide **one** of (or omit for the bundled `example_pendulum` plugin):

- `plugin_id`: short name under `backend_core/AgentMPC/dynamics/plugins/`
  (e.g. `"example_pendulum"`)
- `plugin_path`: path to a `.py` plugin file
- `source`: full plugin Python source (written to a temp file)

Same plugin contract as Streamlit (`DynamicLoader` / `BaseDynamics`).

### Create body (sketch)

```json
{
  "dynamics": { "plugin_id": "example_pendulum" },
  "options": {
    "max_iterations": 15,
    "prediction_horizon": 12,
    "control_horizon": 4,
    "dt_mpc": 0.02,
    "simulation_time": 3.0,
    "ui_scenario_level": 1,
    "use_ui_graph": true,
    "system_name": "cart_pole_pendulum"
  },
  "user_id": null,
  "project_id": null
}
```

## Tests

```bash
PYTHONPATH=. python -m pytest backend_api/AgentMPC/tests/ -v
```

Tests mock the graph invoke path (no live LLM / full simulation required for
structural checks).

## Merge notes

- Replace `job_store.InMemoryJobStore` with Application `job_store` / DB.
- Register module name `"mpc"` if a shared jobs router is introduced.
- Auth / project scoping: optional `user_id` / `project_id` on create; wire
  platform deps on merge.
