# Assignment — AgentMPC backend

**Owner:** AgentMPC backend developer  
**Module root:** `backend_core/AgentMPC/`  
**Standards reference:** `backend_core/AgentPlant/promptTemplate.yaml` + `AgentPlant/agent.py` (`PromptLibrary`)

---

## 1. Externalize prompt templates to YAML

### Current state
Several agents embed full prompt text as Python string constants and `PromptTemplate(...)` objects inside `.py` files. Example: `agents/critic.py` (`CRITIC_PROMPT_TEMPLATE`, `critic_prompt`). The same pattern appears in other agents (e.g. `actor.py` and peers under `agents/`).

### Team standard
Prompts live in **YAML** next to the module (see AgentPlant):

- File: `promptTemplate.yaml` (or a `prompts/` directory with one file per agent)
- Loaded via a small library helper (AgentPlant: `PromptLibrary` / `prompt_library.get_key(...)`)
- Python code only **formats** placeholders and invokes the LLM — no multi-hundred-line prompt strings in source

### Tasks
- [ ] Inventory every hardcoded prompt in `AgentMPC/agents/*.py` (Actor, Critic, Terminator, Juror, advisory/config agents, validators that call an LLM, etc.).
- [ ] Introduce a prompt layout consistent with the rest of LabCD, e.g.:
  - `backend_core/AgentMPC/prompts/actor.yaml`
  - `backend_core/AgentMPC/prompts/critic.yaml`
  - … or a single `promptTemplate.yaml` with named keys
- [ ] Load templates at agent init (prefer existing `labcd_agents.PromptLibrary` if available; otherwise mirror AgentPlant’s pattern).
- [ ] Keep placeholder names stable (`{...}` / f-string fields already used in `PromptTemplate.format`).
- [ ] Remove inlined prompt bodies from `.py` once YAML is the source of truth.
- [ ] Smoke-test: one Actor → Critic turn still produces parseable structured output.

### Acceptance
- No primary system/user prompt bodies remain as large string literals in agent modules.
- Changing a prompt does not require editing Python control flow.
- Style matches `AgentPlant/promptTemplate.yaml` closely enough that a new teammate can find prompts in one place.

---

## 2. Bugfix: oversized `dt_mpc` (e.g. 1.0 s)

### Report summary
Juror/Actor proposals set **`dt_mpc` far too large** (observed **1.0 s**). With `simulation_time=20` and `Np=20`, the reference trajectory length collapses to ~20 samples and the run errors with:

```text
No simulation steps to run: prediction horizon Np=20 is >= the reference
trajectory length (20 samples, from simulation_time=20.0s / dt_mpc=1.0s).
Increase simulation_time or reduce Np.
```

Impact: unusable horizons, high MSE, parameter plots show `dt_mpc` jumping to 1.0 and sticking.

### Relevant code (starting points)
- `mpc/config.py` — `DataConfig.dt_mpc`, `simulation_time`
- `graph/state.py` — `dt_mpc`, `dt_tuned_by_juror`
- `agents/juror.py` — may update `dt_mpc`
- `agents/actor.py` / `agents/critic.py` — proposals and validation feedback
- Trajectory / sim path that raises “No simulation steps…” (evaluator / controller / trajectory generation)

### Tasks
- [ ] **Clamp** `dt_mpc` to a safe upper bound (recommend **≤ 0.1–0.2 s** by default; document the constant in config).
- [ ] Enforce hard feasibility: **`Np * dt_mpc < simulation_time`** (strictly enough steps for a meaningful horizon; align with whatever the trajectory builder uses for `n_steps`).
- [ ] On any proposal that would trigger “No simulation steps” (or equivalent): **reject / rollback** to last good `dt_mpc` (and related params); feed a clear Critic/Juror message instead of failing the whole iteration silently into garbage MSE.
- [ ] Prefer validating **before** committing Juror updates to `cfg.data.dt_mpc` / graph state.
- [ ] Add a unit or integration test that proposes `dt_mpc=1.0` with `Np=20`, `simulation_time=20` and asserts clamp/reject behavior.

### Acceptance
- Runs no longer persist `dt_mpc ≈ 1.0` under normal agent proposals.
- The “No simulation steps to run…” error does not appear for default demo plants when agents tune `dt_mpc`.
- Failed proposals do not wipe best-so-far parameters.

---

## 3. Bugfix: LangGraph recursion limit

### Report summary
```text
GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition.
```
The whole optimization run is marked **failed** even when partial progress exists (e.g. best MSE ~0.02).

### Relevant code (starting points)
- `graph/workflow.py` — graph compile / invoke, stop conditions
- `graph/state.py` — iteration counters, termination flags
- `agents/terminator.py`, `agents/juror.py`, `agents/convergence.py` — stop / continue decisions
- UI entry that invokes the graph (`frontend_streamlit/agent_mpc_app.py` or `run_agents.py`) — where `recursion_limit` may be set

### Tasks
- [ ] Audit why the graph fails to reach a **terminal node** within 25 steps (missing edge, always-continue Juror, no max-iteration stop, etc.).
- [ ] Either:
  - raise `recursion_limit` to a safer value (**50–100**) **and/or**
  - implement a **guaranteed stop**: max iterations, Terminator force-exit, or convergence fallback.
- [ ] **Graceful degradation:** on recursion limit (or max steps), return **best-so-far parameters + warning**, not a hard failure that discards the run.
- [ ] Log a structured reason (`max_iterations` | `recursion_limit` | `terminator`) for the UI activity feed.

### Acceptance
- Typical tuning runs complete or end with best-so-far without an unhandled `GraphRecursionError`.
- Partial progress (best MSE / best params) is preserved and surfaced to the caller.
- Document the chosen limit and stop policy in a short comment in `workflow.py`.

---

## Priority

1. **dt_mpc clamp / horizon feasibility** (correctness of every run)  
2. **Recursion / termination** (reliability of long runs)  
3. **YAML prompts** (maintainability; can follow once runtime is stable)
