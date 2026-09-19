# Adaptive benchmark

The Adaptive benchmark is independent from the MPC benchmark while its format
is being developed and checked.

## Artifacts

`artifacts/` contains Adaptive-format versions of five MPC plants:
cart-pole, double integrator, mass-spring-damper, planar drone, and two-mass spring,
plus a single-pole plant. Each file has a numerical `dynamics(t, x, u)` function,
matching symbolic state equations, Adaptive outputs, and one `test_1`. Each
test stores a full initial-state vector but references only the declared
Adaptive outputs. These output reference values come from the corresponding
MPC targets, where applicable. Tests also add small explicit uncertainty and
disturbance terms to state-derivative equations. The cart-pole
test starts the cart at 0.05 m with zero speed and the pole at 0.02 rad from
upright, also with zero speed.
The cart-pole tracks only pole angle; the drone tracks only altitude and pitch.
Their untracked state components have no reference values.

Each artifact has one manually assigned `total_complexity` on the same discrete
1–10 scale used by the MPC artifacts. The score considers the plant structure,
the selected test's initial state and output references, and its uncertainty and
disturbance. It is an ordinal benchmark label, not a measured control result:

| Artifact | Complexity | Main reason |
| --- | ---: | --- |
| Double integrator | 2 | Simple two-state plant and short position move; mild perturbations. |
| Mass-spring-damper | 3 | Stable two-state plant, but tracking a nonzero spring offset needs sustained force. |
| Single pole | 4 | Two-state nonlinear, unstable upright target; near-upright start and small perturbations. |
| Two-mass spring | 5 | Four coupled states and two tracked positions with different target offsets. |
| Cart-pole | 6 | Nonlinear, unstable, underactuated four-state plant, though only pole angle is tracked and the start is near upright. |
| Planar drone | 7 | Six-state nonlinear two-input plant, altitude change and pitch regulation, with hover thrust and perturbations. |

## Batch benchmark and finalization

Edit the settings at the top of `adaptive_benchmark.py`. `ARTIFACT_FILES` selects
plants (`None` selects all); each selected artifact contributes every test in
`tests.combinations`. `USE_MOCK_AGENTS=True` ignores `LLM_CONFIGS` and makes no
API calls. Set it to `False` to run each listed model; provide its API key via
the repository `.env` or the process environment. Start with one artifact and
one model when using paid APIs. The script validates artifacts before making
API calls, but a failed design or simulation is still recorded as an error so
the other tests can continue. It never changes the artifacts.

```powershell
.venv\Scripts\python.exe benchmarks\Adaptive\adaptive_benchmark.py
.venv\Scripts\python.exe benchmarks\Adaptive\finalize_results.py
.venv\Scripts\python.exe benchmarks\Adaptive\plot_dashboard.py
```

The runner writes `json_reports/results/<model>/batch_tuning_results.json` as
an array of full test records: applied specification, tuning history and best
parameters, agent messages, token usage, and final verification metrics. By
default it omits long time-series arrays. A rerun replaces that model's prior
batch file. The finalizer does **not** tune or call an API; it preserves the full
file and writes `batch_tuning_results_brief.json` beside it. Set
`USE_MOCK_RESULTS=True` to finalize only mock, or set it to `False` and edit
`MODEL_FOLDERS_TO_UPDATE` for real-model reports.

Each brief record includes the artifact's manual `total_complexity` and a
continuous `success_rate` in `[0, 1]`, calculated from final verification:

`success_rate = exp(-max_i(sqrt(tracking_mse.full[i]) / task_scale[i]))`.

`task_scale[i]` is the RMS error that output `i` would have if it stayed at its
initial value throughout the task. The worst normalized output determines the
score, so a good output cannot hide a bad one. A numerically unhealthy or
unbounded run scores zero; absent verification data gives `null`. This uses
full-window tracking (including the transient), unlike Adaptive's binary
`verification.success`, which checks numerical health and a steady-state RMS
threshold. MPC success scores use state-bound widths, so the two modules'
success rates are not directly interchangeable.

`plot_dashboard.py` reads the `_brief.json` files. Its first panel groups mean
success rates by discrete complexity, the second shows mean total tokens per
model, and the third shows mean estimated cost per model. Set
`USE_MOCK_RESULTS=True` to plot only mock, or `False` and edit
`MODEL_FOLDERS_TO_PLOT` for real LLMs. It saves `plots/adaptive_dashboard.png`.
Use `--no-show` to save the PNG without opening a window. Clicking a success
bar in an interactive plot reveals the plant, test, initial state, output
references, and full-window RMSE. Mock costs are synthetic; real-model costs
are estimated using the repository's local price table and recorded tokens.

`artifacts/single_pole.json` contains a Plant Agent output (`python_code` and
`metadata`) plus one test, `test_1`. The initial state is
`[0.05, 0.0]`, ordered as angle and angular velocity, and references only
`theta = 0.0`. Angles are in radians; zero is upright. Uncertainty and disturbance
are small, explicit additions to angular acceleration. The notebooks convert
this artifact into the controller's native `system_spec` and apply the selected
test and simulation settings. In both single-pole and cart-pole, angle zero
radians means upright.

The older `make_tests.py` and `score_complexity.py` scripts describe the previous
multi-plant artifact format; they are not part of these walkthroughs. Do not
run `make_tests.py` on the current artifacts: it writes MPC-style targets and
scenario fields instead of Adaptive output references.

## Mock agents

Both notebooks default to `USE_MOCK_AGENTS=True` and `MOCK_SEED=42`.
Mock mode replaces the Designer, Tuner, and Report Writer responses while
running the actual controller design, simulations, metrics, and best-parameter
selection. It requires no API key. The Designer chooses a method from the
plant structure, the Tuner makes seeded random gain proposals, and the Report
Writer summarizes the numerical outcome. Mock proposals do not guarantee
better control performance. Normal early stopping still applies.

Set `USE_MOCK_AGENTS=False` to use the configured Groq models. This switch is
independent of `ENABLE_TUNING`, which controls whether the tuning loop runs.
The confirmed artifact skips Clarifier chat in both modes; there is no mock
chat Clarifier. The package entry point, CLI, and API service still import
the real pipeline. The notebooks explicitly choose the module to import.

| Mock agent | Original LLM agent |
| --- | --- |
| `designer_agent.py` | `designer_agent_main.py` |
| `tuner_agent.py` | `tuner_agent_main.py` |
| `report_writer.py` | `report_writer_main.py` |

The original LLM factory, pricing, Clarifier, and controller code are unchanged.

For direct Python use, select the import before calling the pipeline:

```python
if USE_MOCK_AGENTS:
    from backend_core.AgentAdaptive.agents.tuner_agent import run_full_pipeline
    agent_options = {"mock_seed": 42}
else:
    from backend_core.AgentAdaptive.agents.tuner_agent_main import run_full_pipeline
    agent_options = {}

result, usage, history, best = run_full_pipeline(
    description="",
    system_spec=applied_spec,
    enable_tuning=True,
    tuning_objectives={"control_effort": 3},
    **agent_options,
)
```

Mock reports are saved under `json_reports/results/mock/`. They include mock
agent messages and usage with `simulated=True`. Input/output token counts are
generated test data. `cost_usd` is a synthetic test cost at $0.001 per token;
`actual_cost_usd` is zero. Final simulation metrics are not fabricated.

The three mock agent files share seeds and synthetic usage helpers from
`backend_core/AgentAdaptive/agents/mock_random.py`.
Offline regression checks (from the repository root):

```powershell
.venv\Scripts\python.exe -m unittest discover -s backend_core/AgentAdaptive/tests -p test_mock_agents.py -v
```

## Result guide and JSON reports

`adaptive_results_guide.pdf` is the permanent A4 guide for understanding the
fields in a full Adaptive result: metrics, tuning history, agent messages,
token usage, best-round selection, and final verification. It can be rebuilt
from a representative result with:

```powershell
.venv\Scripts\python.exe benchmarks\Adaptive\generate_result_guide.py `
  benchmarks\Adaptive\json_reports\results\mock\batch_tuning_results.json
```

Each walkthrough run saves two JSON files beside one another:

- `batch_tuning_results.json` is the complete result.
- `batch_tuning_results_brief.json` contains only the dynamics and test identity, best
  parameters, final verification, token counts, and pipeline status.

The filename is fixed within each model folder, as in the MPC benchmark. A new
run for the same model replaces its previous pair of JSON reports; copy the
files elsewhere first if you want to retain an earlier run.

The walkthroughs do not create a separate PDF for every run.

## Complete tuning example

### Notebook

For a shorter, site-aligned example, open `adaptive_walkthrough_simple.ipynb`.
Its six steps select an artifact/test, set Streamlit-level inputs, build the
confirmed spec, run Groq, save compact JSON, and plot the final response.
Internal `run_full_pipeline` arguments use backend defaults. The selected
priority is control effort (importance 3); set priorities to `None` to use the
default selection. The selected test's uncertainty/disturbance entries are used
unless explicitly replaced by user-confirmed terms. Reports are saved
under `json_reports/results/groq-simple/`. No Clarifier chat is executed, so
its token costs are excluded. Mock mode saves under the mock folder instead.

Open `adaptive_walkthrough.ipynb` using the repository's `.venv` Python kernel
and execute cells in order. It reads `artifacts/single_pole.json` and
selects `test_1` by default. With mock mode disabled it uses Groq
(`openai/gpt-oss-120b`) for the Designer, Tuner, and Reporter roles.
For real runs, put `GROQ_API_KEY` in the repository `.env`
or your environment; no credentials are printed or saved in reports.

Editable settings include the artifact/test ID, role models/token limits,
simulation time/step, tuning objectives, target error, and maximum tuning rounds.
The selected test already includes small uncertainty and disturbance, which
activate Adaptive estimation. Optional example-effect settings replace those
terms in memory. The artifact file is
never modified. The Adaptive runner has no numbered scenario-level preset;
initial state, reference, uncertainty, and disturbance are supplied explicitly.

`RUN_TUNING=True` runs the selected agent mode, saves a report under
`json_reports/results/mock/` or `json_reports/results/groq-gpt-oss-120b/`,
and plots the final response and tuning
history. A new run replaces the selected model's previous results.
`RUN_TUNING=False` reads `RESULT_FILE` without API calls. The saved spec must
match the selected settings.
`ENABLE_TUNING=False` skips only the Tuner; with real agents, Designer and
Reporter still make API calls.
An initially adequate design can also skip extra tuning rounds automatically.

`SAVE_TIME_SERIES=False` keeps JSON compact; the current run can still plot its
in-memory trajectory. Use `True` to retain trajectories for plotting saved
reports later. The plotted tracking percentage is Adaptive's native tracking
metric, not MPC's exponential success score. All public `run_full_pipeline`
arguments are supplied explicitly. Seed gains are not a public argument to
that function; the existing backend supplies the initial controller gains.

### Python example

`adaptive_tuning_test.py` is a standalone learning example. It defines
a numerical Python double-integrator plant, converts the matching AgentPlant
metadata to an Adaptive `system_spec`, configures all public tuning-run inputs,
uses Groq for the Design/Tuner/Reporter roles, and saves its result under
`Adaptive/results/`.

Add `GROQ_API_KEY` to the repository `.env`, then run:

```powershell
.venv\Scripts\python.exe benchmarks\Adaptive\adaptive_tuning_test.py
```
