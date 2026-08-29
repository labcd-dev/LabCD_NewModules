# Assignment — AgentAdaptive backend

**Owner:** AgentAdaptive backend developer  
**Module root:** `backend_core/AgentAdaptive/`  
**Shared LLM utilities:** `packages/labcd_agents` (`labcd_agents`)

---

## 1. Route all LLM cost/pricing through `labcd_agents`

### Current state
AgentAdaptive keeps a **local** pricing module:

- `backend_core/AgentAdaptive/tools/model_pricing.py`  
  - Owns `MODEL_PRICING`, `AVAILABLE_MODELS`, `cost_for`, `format_run_cost_report`, UI labels, etc.

Call sites include:

| Location | Usage |
|----------|--------|
| `agents/tuner_agent.py` (~line 780) | `model_pricing.format_run_cost_report(usage)` after a pipeline run |
| `tools/pdf_report.py` | `run_cost_rows`, `format_cost_plain` for report tables |
| `frontend_streamlit/streamlit_app.py` | Model pickers (`AVAILABLE_MODELS`, `INHERIT_LABEL`, `model_label`) and run cost display |

### Team standard
All **LLM-related business logic** (providers, tokens, **pricing/cost**) belongs in the shared package:

- Reference implementation: `packages/labcd_agents/src/labcd_agents/pricing.py`  
  - `ModelPrice`, `DEFAULT_PRICE_TABLE`, `CostCalculator`  
  - Resolve model ids, compute USD cost from input/output tokens  
- Related: `labcd_agents.tokens` for usage normalization

Agent modules should **not** maintain parallel price tables or cost formatters that drift from the rest of LabCD.

### Tasks
- [ ] Map every public function/constant in `tools/model_pricing.py` to an equivalent (or thin adapter) on `labcd_agents.pricing.CostCalculator` / package APIs.
- [ ] Extend `labcd_agents.pricing` **only if needed** (e.g. cached-input rates, `format_run_cost_report`, inherit label) so Adaptive does not keep a second source of truth. Prefer enhancing the shared package over leaving Adaptive-specific pricing logic behind.
- [ ] Update:
  - `agents/tuner_agent.py` — replace `model_pricing.format_run_cost_report(...)`
  - `tools/pdf_report.py` — cost rows / formatting
  - `frontend_streamlit/streamlit_app.py` — model list, labels, cost table (coordinate with UI if signatures change)
- [ ] Deprecate or delete `tools/model_pricing.py` once call sites are migrated (or shrink it to a short re-export shim with a deprecation comment, then remove in a follow-up).
- [ ] Ensure price table coverage still includes models Adaptive exposes in the Streamlit UI (merge any Adaptive-only ids into `DEFAULT_PRICE_TABLE` or register via `CostCalculator`).
- [ ] Smoke-test: one Adaptive pipeline run still prints/returns a sensible cost summary; PDF cost section still renders.

### Acceptance
- No independent `MODEL_PRICING` dict remains as the system of record inside AgentAdaptive.
- Cost numbers for a known `(model, input_tokens, output_tokens)` match `labcd_agents.pricing.CostCalculator`.
- Tuner / Streamlit / PDF all import pricing from `labcd_agents` (or a one-line Adaptive facade that only wraps `labcd_agents`).

### Out of scope for this item
- Changing Clarifier / Designer / Tuner control logic unrelated to billing
- Frontend mockup HTML (`frontend_mockup/`)
- AgentMPC prompts or graph work (separate assignment under `AgentMPC/`)

---

## 2. Harden SMC relative-degree search (silent hang on bad plants)

### Issue
When a plant’s declared output is **not influenced by any control input** (undefined / infinite relative degree), the Adaptive design stage can hang indefinitely. The Streamlit UI shows an open-ended progress message such as:

```
Running the full simulation… (300s+)
```

with no exception and no stage failure — a silent failure from the user’s point of view.

Root cause is **not** a slow numeric integrator. Open-loop integration of a typical plant finishes in tens of milliseconds. The hang is deterministic and lives in the SMC structure builder:

- **File:** `controller/smc_design.py` → `_build_smc_structure`
- **Mechanism:** unbounded `while True` that searches for relative degree by successive Lie derivatives of each output until an input appears:

```python
while True:
    ...
    input_present = any(derivative.has(u) for u in inputs)
    relative_degree += 1
    if input_present:
        break
    derivatives.append(derivative)
```

If the input never appears in any derivative, the loop never terminates.

**Illustrative plant** (structurally uncontrollable from the declared output):

| state | equation |
|-------|----------|
| `v_in` | `0` (frozen exogenous state) |
| `v_out` | `(v_in - v_out) / R` |
| `i_L` | `(v_in * d - v_out) / L` |
| output | `y = v_out` |

Because `f[0] ≡ 0`, every higher Lie derivative of `v_out` still lacks the input `d`. Relative degree is infinite → the loop never breaks → silent hang (no exception, so the pipeline worker never surfaces an error).

The same class of failure occurs for any plant where:

- a state that multiplies the control has zero dynamics and a zero (or constant) value that keeps the control term dead, or
- the control input is structurally absent from all Lie derivatives of the chosen outputs (e.g. wrong output choice, incomplete equations from an upstream plant model).

### Tasks
- [ ] Add a hard `max_relative_degree` guard (suggested default: 6) in `_build_smc_structure`. If the bound is hit, raise a clear `ValueError` that names the output and states that the plant is not controllable from that output (or has undefined relative degree).
- [ ] Propagate the error so `runs._run_smc` / the pipeline emits a failure / `stage_done` event with the message instead of hanging; the Streamlit UI should then show the error in the trace.
- [ ] Optionally: same guard in the backstepping path if it performs an analogous unbounded search.
- [ ] Add a unit test that feeds a structurally uncontrollable plant (e.g. the table above, or a minimal `ẋ = 0`, `y = x`) and asserts a fast, explicit failure rather than a timeout.
- [ ] (Plant side, coordinate with AgentPlant / PlantCompiler) Prefer models where control inputs reach the declared outputs; treat “input never appears in any Lie derivative” as a validation warning when compiling adaptive_spec.

### Acceptance
- Uncontrollable / infinite-relative-degree plants fail within a few seconds with an actionable error message.
- Controllable plants (existing Adaptive demos) are unchanged.
- Streamlit Adaptive run no longer shows an open-ended “Running the full simulation…” spinner for this class of plant.

### Out of scope for this item
- Automatically repairing bad plant equations
- Changing default SMC gains or RBF settings

---


---

## 3. Design final build/simulation: `TypeError` with complex numbers (overly graceful fallback)

### Symptom (UI)
Under **Design Agent — Summary** (`frontend_streamlit/streamlit_app.py`, ~line 1239 — user-facing string `Design Agent — Summary`; historically also referenced near adaptive app summary panels):

```text
Design passed review but failed during the final build/simulation:
TypeError: float() argument must be a string or a real number, not 'complex'
```

- **No response chart** is produced.
- A **PDF is still generated** — fallback is too graceful: the run looks partially successful while the numeric design is unusable.
- Message origin (backend): `agents/tuner_agent.py` (~line 747):
  `"Design passed review but failed during the final build/simulation: ..."`.

### Likely failure class
Somewhere in the **post-review final build / simulation** path (`controller/runs.py` → `simulation.simulate` / structure build / scoring), a value becomes **complex** (e.g. fractional powers, `sqrt` of a negative intermediate, or sympy/numeric mix) and is passed to `float(...)`.

This is distinct from the infinite-relative-degree hang (section 2): here an exception is raised, but the pipeline **swallows it into a soft summary + PDF** instead of failing the design stage cleanly.

### Reproducing artifact
Full artifact used when the error was observed (`Inverted-Pendulum_ed1bc9`). Persist under the project artifact store or reload via Adaptive “Load from artifact”:

```json
{
  "artifact_id": "Inverted-Pendulum_ed1bc9",
  "system_name": "Inverted Pendulum",
  "created_at": "2026-08-29T08:42:35.353522+00:00",
  "version": "1.0",
  "plant": {
    "python_code": "import numpy as np\n\ndef dynamics(t, x, u):\n    theta = x[0]  # angle (rad)\n    omega = x[1]  # angular velocity (rad/s)\n    g = 9.81  # gravity (m/s^2, assumed)\n    m = 0.5  # mass (kg, assumed)\n    l = 1.0  # length (m, assumed)\n    I = (1/3) * m * l**2  # moment of inertia (kg*m^2)\n    tau = u[0]  # input torque (N*m)\n    dtheta = omega  # angular position derivative\n    domega = (tau - m * g * l * np.sin(theta)) / I  # angular velocity derivative\n    dx = np.array([dtheta, domega])\n    return dx",
    "metadata": {
      "states": [
        "theta",
        "omega"
      ],
      "state_meanings": [
        "angle (rad)",
        "angular velocity (rad/s)"
      ],
      "inputs": [
        "tau"
      ],
      "outputs": [
        "theta"
      ],
      "state_equations": [
        "omega",
        "(tau - m * g * l * sin(theta)) / I"
      ],
      "parameters": {
        "g": 9.81,
        "m": 0.5,
        "l": 1.0,
        "I": 0.16666666666666666
      },
      "system_type": "SISO",
      "assumptions": [
        "g = 9.81 m/s^2 (assumed)",
        "m = 0.5 kg (assumed)",
        "l = 1.0 m (assumed)"
      ]
    }
  },
  "pre_launch": {
    "total_simulation_time": 10.0,
    "solver_sample_time": 0.001,
    "initial_state": [
      0.1,
      0.0
    ],
    "default_target": [
      0.0,
      0.0
    ],
    "trajectory_mode": "sin",
    "trajectory_amplitude": 0.69,
    "trajectory_frequency": 0.333,
    "trajectory_offset": 0.09
  },
  "module_specific": {
    "adaptive": {
      "clarifier_record": null,
      "designer_method": null,
      "tuning_best": null
    },
    "mpc": {
      "suggested_dt": null,
      "suggested_Q": null,
      "suggested_R": null,
      "suggested_feedforward": null,
      "derivative_pairs": null
    }
  }
}
```

**Notes on this plant:** simple inverted-pendulum-style 2-state model; `pre_launch` uses **sin** reference (`amplitude=0.69`, `frequency=0.333`, `offset=0.09`). Prefer replaying with this artifact so Designer → final build hits the same path.

### Tasks
- [ ] Reproduce with the artifact above (or equivalent `system_spec` from `PlantCompiler.generate_adaptive_spec`).
- [ ] Locate the `float(...)` call (or numpy cast) that receives a complex value; common areas: `controller/structure_build.py`, `controller/simulation.py`, `controller/runs.py`, design codegen / lambdify.
- [ ] Fix root cause (keep expressions real-valued; reject or rewrite steps that introduce complexes; validate numeric coeffs before sim).
- [ ] **Tighten fallback policy:** if final build/simulation raises, mark design stage **failed**, do **not** present a success-tinted summary, and **do not** emit a “successful” PDF without plots/metrics (or clearly stamp the PDF as **FAILED — no simulation**).
- [ ] Surface the underlying exception type + message in the Design Summary and pipeline events (keep user-readable text, but fail closed).
- [ ] Add a regression test that either runs this plant through final build or unit-tests the offending conversion with a complex input and expects a controlled failure.

### Acceptance
- Replaying `Inverted-Pendulum_ed1bc9` either **simulates with charts** or **fails loudly** with a clear error — not “passed review” + empty plots + PDF.
- No silent `TypeError: ... not 'complex'` hidden behind a soft Design Summary line.
- PDF generation policy documented: only full success, or explicit failure report.


## Notes for the implementer

- `tuner_agent.py` builds a structured `usage` object (per-stage tokens + timeline) before calling `format_run_cost_report`. Preserve that reporting shape for the UI unless you intentionally redesign the cost panel and update Streamlit accordingly.
- If `labcd_agents` lacks cached-input pricing, either add it to `ModelPrice` / `CostCalculator` in the package or document a deliberate simplification — do not silently double-bill or drop costs without a comment.
- The relative-degree hang is independent of the pricing work; it can be fixed in a separate PR.
