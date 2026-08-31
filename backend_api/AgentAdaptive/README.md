# AgentAdaptive FastAPI

Job-oriented HTTP API for adaptive / nonlinear control design
(Clarifier → Designer → build/sim → Tuner → Report Writer).

Core logic lives in `backend_core/AgentAdaptive/`. This package is a thin
adapter + in-memory job store, matching the layout of `backend_api/AgentPlant/`.

## Run

```bash
# from repository root
PYTHONPATH=. uvicorn backend_api.AgentAdaptive.app:app --host 127.0.0.1 --port 8004
# or
PYTHONPATH=. python -m backend_api.AgentAdaptive.app
```

Health: `GET http://127.0.0.1:8004/health`

API prefix: `LABCD_API_PREFIX` (default `/api`).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/adaptive/jobs` | Start a design job (`system_spec` + options) |
| `GET` | `/api/adaptive/jobs` | List jobs |
| `GET` | `/api/adaptive/jobs/{job_id}` | Status + stage + progress |
| `POST` | `/api/adaptive/jobs/{job_id}/clarify` | Submit clarification answer or `force_finish` |
| `POST` | `/api/adaptive/jobs/{job_id}/cancel` | Request cancel |
| `GET` | `/api/adaptive/jobs/{job_id}/results` | Final artefacts (report, metrics, tuning, usage) |

### Job lifecycle

1. **Submit** with a `system_spec` (plant dynamics + sim knobs).  
   - Default: job enters **clarifying**; poll status and answer via `/clarify`.  
   - `options.skip_clarify: true` → design pipeline starts immediately.
2. **Clarify** until the model returns `complete` (or client sends `force_finish`).
3. Pipeline runs in a background thread: design → build → optional tune → report.
4. Poll `GET .../jobs/{id}` for `status` / `stage` / `progress`.
5. When `completed` (or `failed` / `cancelled`), fetch `GET .../results`.

### Create body (sketch)

```json
{
  "system_spec": {
    "status": "complete",
    "system_name": "cart_pole",
    "dynamics": { "states": [], "inputs": [], "...": "..." }
  },
  "options": {
    "enable_tuning": false,
    "target_rms_frac": 0.02,
    "max_tuning_rounds": 4,
    "skip_clarify": false
  }
}
```

`system_spec` can come from AgentPlant `GET /api/plant-model/artifacts/{id}/adaptive-spec`.

## Tests

```bash
PYTHONPATH=. python -m pytest backend_api/AgentAdaptive/tests/ -v
```

Tests mock Clarifier and `run_full_pipeline` (no live LLM required).

## Merge notes

- Replace `job_store.InMemoryJobStore` with Application `job_store` / DB.
- Register module name `"adaptive"` if a shared jobs router is introduced.
- Auth / project scoping: optional `user_id` / `project_id` on create; wire platform deps on merge.

## Simulation series (charts)

Completed jobs may include a `series` object on `GET /api/adaptive/jobs/{id}/results`
for frontend charting (e.g. Recharts):

- `channels.t` — time vector
- `channels.y` / `ref` / `u` / `x` — time-major arrays with `names`

Downsampling: default max 2000 points (`LABCD_ADAPTIVE_SERIES_MAX_POINTS`).
Metrics remain full-resolution.

Interactive matplotlib is **off** for API workers (`LABCD_ADAPTIVE_SHOW_PLOTS=0`,
`MPLBACKEND=Agg`). Set `LABCD_ADAPTIVE_SHOW_PLOTS=1` for local GUI plots.
