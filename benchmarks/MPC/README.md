# MPC benchmark

The MPC scripts follow the same organization as the Adaptive scripts: settings
at the top, selection and validation helpers, one-test execution or report
processing, and a `main()` entry point. Core controller and agent code is not
modified.

## Workflow

From the repository root, using the project's Python environment:

```powershell
python benchmarks/MPC/mpc_benchmark.py
python benchmarks/MPC/finalize_results.py
python benchmarks/MPC/plot_dashboard.py
```

| Script | Responsibility |
| --- | --- |
| `mpc_benchmark.py` | Select models/artifacts, tune each test, verify the best parameters, checkpoint accepted-format JSON |
| `finalize_results.py` | Match saved runs, migrate legacy scales once, and update complexity plus both success scores in place |
| `plot_dashboard.py` | Create two dashboards: one for each success score, with mean tokens/cost |
| `plot_trajectories.py` | Plot saved verification states with their targets, and applied control inputs versus time |

Importing these modules does not run tuning or write a report. Only the benchmark
can call an LLM or run a simulation. Finalization and plotting are offline.

## Benchmark settings

Edit `USER SETTINGS` near the top of `mpc_benchmark.py`:

- `RUN_BENCHMARK = False` disables execution.
- `ARTIFACT_FILES = None` selects all active artifacts. A filename list selects
  a subset in the specified order.
- `USE_MOCK_AGENTS = True` uses `benchmarks/MPC/mock/` and ignores the real-model
  list. Controller evaluation and verification still use real simulations.
- With mock mode off, `LLM_CONFIGS` defines the model, provider, API-key
  environment variable, and results folder for each enabled model.
- `CHEAPEST_MODEL_ONLY = True` selects the entry with the smallest sum of
  input/output token rates in the local price table. This uses equal estimated
  input/output counts, not a guarantee of the cheapest actual run.
- `MAX_ITERATIONS`, `MIN_EXPLORE_ITERATIONS`, `EXPLORATION_INTENSITY`,
  `USER_GUIDANCE`, and optional `SEED_PARAMS` control tuning.
  With `SEED_PARAMS = None`, the Actor proposes the first parameters.
- `RUN_SEEDS = [42, 123, 2024]` repeats every selected model/artifact/test three times.
  Every repetition starts a fresh graph, controller and token tracker. Its seed
  is supplied to the LLM client, simulation RNG and (in mock mode) mock agents.
  Seeds do not change declared initial conditions, targets, bounds or cost
  weights. With nominal/noiseless artifacts, simulation seeds alone introduce
  no randomness; real tuning variation comes from the agents' proposals.
  Provider responses are not guaranteed to be exactly reproducible.
  Three seeds mean three times as many tuning runs and potentially API usage.

`USER_GUIDANCE` is free-text direction included in the agents' prompts (for
example, emphasizing tracking or control effort). It guides proposals and
feedback; it does not replace the numeric cost weights. An empty string uses
the agents' default balanced guidance.

`SEED_PARAMS` is an optional initial MPC parameter guess: `Np`, `Nc`, `Q`, `R`,
`P`, and optionally `dt`. With a guess, the graph evaluates it before the Actor
starts revising parameters. It does not fix those parameters permanently and
is not a random-number seed. `Q`/`P` need one weight per state and `R` one per
input, so a single guess must suit every selected plant. Leaving it `None`
lets the Actor construct a dimensionally appropriate initial proposal.

Simulation settings are read separately for every artifact:

```json
"simulation_parameters": {
  "simulation_time": 20.0,
  "dt_simulation": 0.005,
  "dt_mpc": 0.01,
  "settling_tolerance": 0.05,
  "cost_weights": {
    "mse": 2.0,
    "overshoot": 0.1,
    "settling_time": 0.1,
    "control_effort": 0.5,
    "oscillation": 0.0,
    "unsettled_penalty": 1.0
  }
}
```

These are validated before LLM setup. An Actor proposal can subsequently change
`dt_mpc`; verification uses the effective `dt` of the best candidate.
Each artifact's `cost_weights` supplies the composite-cost coefficients used
to rank tuning candidates. These are not the controller's state/input/terminal
weights `Q`, `R`, and `P`, which the Actor tunes. The nine artifacts initially
retain the previous shared coefficients, so moving the settings does not itself
change the objective. Keys must be supported and values finite and nonnegative.
Set `cost_weights` to JSON `null` to request the core's balanced defaults;
omitted individual keys in an object use the core's defaults for those terms.
Artifact `fixed_targets` supply Success Rate 2's offline pass/fail limits.
They do not alter the core's tuning termination rules or cost weights.

## Artifact structure

Active artifacts are `AUV.json`, `AircraftPitch.json`, `BallBeam.json`,
`CSTR.json`, `CartPole.json`, `DCMotor.json`, `InvPendulum.json`,
`QuanserHelicopter.json`, and `TwoLinkManipulator.json`.

Each artifact contains:

- `system_name` and `python_code`, defining `dynamics(t, x, u)`.
- `metadata`, including ordered `states`, `inputs`, `outputs`, equations,
  parameters, and assumptions.
- `simulation_parameters`.
- `tests.combinations`, each containing `test_id`, `total_complexity`,
  `initial_condition`, `target`, and native MPC scenario settings.

State/input counts come from the metadata lists. IC and target each contain
one value per state. Unlike Adaptive output references, these are full-state
MPC references; this structural refactor does not change which states the MPC
cost penalizes.

Each test's `total_complexity` is a manually assigned integer from 1 to 10, or
`null` until rated. It is not calculated from plant/condition components.

The benchmark-local `compile_artifact_dynamics()` converts the embedded function
to the interface required by AgentMPC. `build_mpc_config()` constructs a fresh
configuration for each test. Neither changes the core. The adapter passes
`metadata.state_bounds` and `metadata.input_bounds` into the plugin's
`SystemConfig`, which the core controller already reads. Each field is either
`null` (unbounded) or `[lower_vector, upper_vector]` in metadata state/input
order. Vectors must contain finite numbers with lower <= upper; IC/target must
lie inside state limits and the starting trim must lie inside input limits.
Validation runs before LLM setup. The active artifacts omit derivative pairs.
Aircraft/CSTR trim metadata, where present, remains the adapter's equilibrium
input; this refactor does not alter its existing behavior.

Input limits are copied from `py files/jsons/*` (`min_ctrl`/`max_ctrl`):
AircraftPitch `[0, -25]` to `[1, 25]`, AUV/BallBeam `-0.5` to `0.5`,
DCMotor `-24` to `24`, CSTR `-3` to `0`, InvPendulum `-1` to `1`,
QuanserHelicopter `-24` to `24` per motor, and TwoLinkManipulator `-1` to `1`
per joint. Merged models retain both channels' limits. CSTR's input is signed
as in the source (`physical cooling flow = -cooling_input`); AircraftPitch's
elevator input is in degrees, not radians. These are absolute applied-input
limits, not limits on input changes or trim-relative deviations.
The provided sources specify no state limits, so all `state_bounds` remain
`null`. There is no CartPole source in `py files`, so its `input_bounds` is
also `null`; no limits were inferred from ICs, targets, or physical parameters.
Tight limits can make targets difficult or infeasible and change future runs;
existing saved results are not recomputed. State limits constrain linearized
predictions and also trigger plugin termination if actual simulated states
leave the bounds; they are not a guarantee of nonlinear trajectory safety.
The separate PlantCompiler has not been changed; this connection is local to
the benchmark adapter.

Artifact Python is executable code: only run trusted artifacts. Equations and
manual complexity ranks are benchmark definitions, not independently validated
reproductions of published plants or difficulty measurements.

## Scenarios

The current tests use native Level 1 Nominal: no injected state noise or physical
parameter perturbation. Levels 2 and 3 remain available through
`apply_scenario_level()`; they are not Adaptive uncertainty/disturbance
expressions. Scenario behavior is unchanged by the refactor.

## Reports and finalization

### Run progress

`json_reports/results/benchmark_progress.json` tracks the currently selected
models, seeds, artifacts and tests. It contains completed and remaining
combination lists, the running combination, failures, and a summary such as
`"done_out_of_total": "27/108"`. Failed attempts also remain in the remaining
list. "Done" means tuning and verification finished and the result JSON was
saved; it does not mean the controller achieved a high success score.

The benchmark checks matching saved seeded results at startup, then updates
the tracker before and after each test and on interruption. Disabled models
and unselected artifacts are not counted. Unreadable saved reports are marked
as pending with an explanation. Matching checks plant identity, seed, model,
test ID, initial condition, target and scenario level, but do not certify that
every historical tuning setting is unchanged.

This is a progress inventory, not automatic resume: running the benchmark
still reruns every selected combination and replaces its existing report.
The tracker describes the latest selection; it is not an archive of all past
benchmark selections. Controller/agent core code is unchanged.

### Saved reports

Each model/seed writes into `json_reports/results/<folder_name>/seed<value>/`:

```text
json_reports/results/gpt-4.1-nano/
    seed42/
        AUV_results.json
        AircraftPitch_results.json
        trajectories/
    seed123/
        ...
    seed2024/
        ...
```

Folders use the actual seed value: `RUN_SEEDS = [42, 43, 44]` produces
`seed42`, `seed43`, `seed44`. Model selection remains controlled by
`LLM_CONFIGS` and `CHEAPEST_MODEL_ONLY`; the example above does not select nano.

- `<Artifact>_results.json`, for example `AircraftPitch_results.json`: compact
  tuning summary and best-parameter verification for that artifact's tests.
  Complexity and success start as placeholders; finalization fills them in the
  same file. No `_full` copy is created.

Each file is a JSON array with one record per test, even if the artifact has only
one test. The benchmark checkpoints after each test inside that artifact's file.
Re-running a selected artifact replaces that model/seed/artifact's base report; other
artifact reports remain untouched. New reports follow the structure documented
in `json_reports/accepted_results_template.json`; its explanation strings are
documentation, not literal values copied into results:

- `seeds`: the repetition seed, LLM seed (null for mocks), simulation seed,
  and mock seed (null for real models).
- `tuning`: best parameters, best cost/MSE, iteration count, termination reason,
  Juror verdict, total token usage, and completion flag. `final_state` and
  `node_outputs` are not saved; workflow memory still exists internally while
  tuning and selecting the controller.
- `tuning.completed`: whether the graph reached its normal end. If a later
  node raises, the last available tuning summary is retained and this flag
  is false. Failed runs also include a top-level error type and message.
- `verification.raw_result`: best-parameter verification metrics, step count,
  solver diagnostics, error details if present, and a `trajectory_file` link.
  The four full-length arrays are stored separately, not inside the JSON.

- `verification.outputs`, `tracking_mse.full`, and `task_scale`: short arrays
  with one entry per declared output, not per simulation timestep.

Duplicate verification `metrics`, `status`, `average_mpc_solution_time_seconds`,
and `solver_diagnostics` summaries are not saved. Their values remain inside
`raw_result`. Finalization and plotting read that location and also accept
older reports with the legacy summary fields. Finalization changes only
`total_complexity`, `success_rate_1`, `success_rate_2` and its check details;
it also converts older RMS
`verification.task_scale` arrays to MSE once and adds `task_scale_basis: "mse"`.
Other fields, including token costs and trajectory links, remain unchanged.
Each test writes `trajectories/<Artifact>_<test_id>_verification.npz` beneath
its model/seed folder. This compressed NumPy archive stores `states`, `inputs`,
`reference`, and `times` with their full samples, shapes and numeric dtypes;
there is no downsampling. Metrics and output summaries are computed from the
complete simulation. `trajectory_file` is relative to the containing results
JSON and is null if no archive is available. Re-running a test replaces its
archive, just as it replaces its model/artifact base report.
Keep the JSON and its `trajectories` folder together when moving results.
Finalization preserves the archive link because the JSON stays in place.
Existing saved reports are not converted automatically. JSON non-finite values
become `null`; numeric NaN/Inf values in the NPZ remain unchanged.

Load a test's complete trajectories for plotting:

```python
import json
from pathlib import Path
import numpy as np

report = Path("benchmarks/MPC/json_reports/results/mock/seed42/AUV_results.json")
record = json.loads(report.read_text(encoding="utf-8"))[0]
trajectory_file = record["verification"]["raw_result"]["trajectory_file"]
if trajectory_file is not None:
    with np.load(report.parent / trajectory_file, allow_pickle=False) as archive:
        states = archive["states"]
        inputs = archive["inputs"]
        reference = archive["reference"]
        times = archive["times"]
    # Plot state/reference columns against times. Control inputs can have one
    # fewer row than states; plot them against times[:len(inputs)].
```

Mock token counts/costs are synthetic; real costs are estimates using the local
price table. Token totals cover the entire tuning run. In mock mode, counts are
generated reproducibly and priced using `MOCK_COST_USD_PER_TOKEN`, while the
report's synthetic usage is explicitly marked `simulated`.

In `finalize_results.py`, configure `USE_MOCK_RESULTS` and
`MODEL_FOLDERS_TO_UPDATE`. Mock mode ignores the list and explicit input paths.
Real mode accepts the configured folders or explicit CLI report paths.
The finalizer discovers all `<Artifact>_results.json` files in each selected
model's saved seed folders and updates them in place. Plotting reads those
same files. Set `SEEDS_TO_UPDATE = [42]` in the finalizer to update only seed42;
`None` selects every saved seed. Set `SEEDS_TO_PLOT = [42]` in each plotter
to plot only seed42; `None` selects all saved seeds. The dashboard averages
success across selected repetitions for each model/plant/test, and averages
tokens/cost per run. Trajectory figures stay separate under
`plots/<model>/seed<value>/<artifact>/`, so seeds cannot overwrite each
other's plots. The trajectory CLI also accepts `--seeds 42 123 2024`.
When a model has saved seed runs, old unseeded reports are ignored to avoid
mixing repetitions with historical data. If none exist, the old model-folder
layout remains supported.
Both retain legacy batch-report support only when no new-format reports are
available for that model, avoiding duplicate counts. Existing batch reports
are not renamed or deleted. Older `_full` copies are ignored when current
reports exist; the dashboard accepts them only as a historical fallback.
When only some artifacts have been rerun in the new
format, the model's dashboard uses the new-format finalized reports, not a mix
of new and older batch data. Finalize new base reports before plotting.
The finalizer rejects results whose IC, target, scenario level, or plant identity
no longer match the current artifact, rather than attaching a new test's score
to an old run. Missing output MSE or normalization data leaves success `null`.

Seed-aware offline commands (omit `--seeds` to use each script's settings;
`None` in those settings means all saved seeds):

```powershell
python benchmarks/MPC/finalize_results.py --models gpt-4.1-nano --seeds 42 123 2024
python benchmarks/MPC/plot_dashboard.py --models gpt-4.1-nano --seeds 42 123 2024 --no-show
python benchmarks/MPC/plot_trajectories.py --models gpt-4.1-nano --seeds 42 123 2024
```

The root `benchmark_progress.json` is not a controller result and is ignored
by finalization and both plotters. Each seed JSON keeps its own success score.
Only the dashboard averages those scores, separately for each LLM/artifact/test:
`mean_seed(success_rate_1)` or `mean_seed(success_rate_2)`. It does not average tracking errors before computing
success, or combine different plants at the same complexity. Null scores are
excluded; valid zero scores are included. A partially available case uses only
its available scores; click its bar to see the contributing seeds and run count.
Duplicate records for the same LLM/case/seed, conflicting complexities, or changed
test conditions raise an error rather than silently biasing the mean.
Saved run seeds must agree with their seed-folder names.

Trajectory plots remain separate per seed, not averaged; their archive links are
resolved relative to their own seed's JSON file. The trajectory plotter's
`SEEDS_TO_PLOT` setting is independent of the dashboard's seed filter.

MPC success is scored only on `metadata.outputs`, not on the aggregate error
of every state. For each declared output `i`, use its full-window MSE and a
mean-squared task scale:

```text
MSE_i = mean_t((actual_output_i(t) - reference_i(t))^2)
task_scale_mse_i = mean_t((initial_output_i - reference_i(t))^2)
success_rate_1 = exp(-10 * mean_i(MSE_i / task_scale_mse_i))
```

`i` identifies an output state, not a tuning round; `t` spans all verification
samples. For constant targets, `task_scale_mse_i = (initial_i - target_i)^2`.
For multiple outputs, average their individual ratios inside the exponential,
not their success rates and not an aggregate MSE divided by an aggregate scale.
For AircraftPitch this scores only `pitch_angle`; for TwoLinkManipulator it
scores both joint angles. This affects reporting only: the MPC controller still
uses full-state references and its existing Q/R/P tuning objective.

The finalizer reads **only** the saved `verification.tracking_mse.full` and
`verification.task_scale` arrays, aligned by `verification.outputs` names to
the artifact's `metadata.outputs`. The benchmark now stores task scales in
MSE units and includes `verification.task_scale_basis: "mse"`. Such denominators
are used directly, without squaring again. Unmarked historical MPC reports
stored RMS scales: finalization squares them once, saves the converted array
in the same JSON, and adds the MSE marker. Repeated finalization is idempotent.
If manually supplying an MSE task scale, include the marker to prevent it from
being interpreted as a historical RMS scale.
Missing summaries leave success `null`; the finalizer does not reconstruct
them from artifact targets, raw per-state metrics, aggregate errors, or NPZ
trajectories. A hard numerical failure still scores 0.

The score is rounded to six decimal places and plotted as a percentage. For zero
task scale on an output, its ratio is zero only when its MSE is also zero;
otherwise the run scores 0.
A numerically unstable/bounds-terminated verification
also scores 0. An unsettled but numerically healthy run is still graded by its
full-window error. This score is separate from the core's boolean success flag.

The benchmark calculates mean-squared task scales from the full saved reference
window. Finalization uses these saved values rather than recalculating them.
Existing reports with output MSE and RMS scale summaries can be migrated and
rescored without rerunning tuning or simulation. No `_full` copies are created.
The full verification trajectories are retained in the NPZ archive;
MPC's tuning objective is unchanged. The dashboard reads metrics from JSON and
does not need to open the NPZ archive.

This formula uses mean-squared quantities; it is not interchangeable with a
formula using RMSE divided by an RMS scale. Each ratio here is the square of
the corresponding normalized-RMSE ratio. Adaptive's formula is unchanged.

### Success Rate 2: fixed-target pass fraction

The two calculators are separate, offline modules:

- `success_rate/success_rate_1.py`: the existing exponential MSE/task-scale formula, unchanged.
- `success_rate/success_rate_2.py`: equally weighted fixed-target checks.

For each saved run:

```text
success_rate_2 = passed_targets / enabled_targets
```

Three passed checks out of four give 0.75 (75%). This is partial-target credit,
not the paper's strict full-task pass rate: 100% requires every enabled target
to pass. `success_rate_2_details.all_targets_passed` records that distinction.

| Artifact limit | Verification comparison |
| --- | --- |
| `mse_max` | Mean of `verification.tracking_mse.full` for the artifact's named controlled outputs <= limit |
| `settling_time_max_s` | `raw_result.metrics.settling_time` <= limit AND `settled` is true |
| `overshoot_max_pct` | 100 × `raw_result.metrics.overshoot` <= limit, with `overshoot_meaningful` true |
| `control_effort_max` | `raw_result.metrics.control_effort` <= limit |

MSE is NOT the aggregate all-state `raw_result.metrics.mse`, RMSE, normalized
MSE or a steady-window error. With multiple controlled outputs, average their
full-window MSEs; each enabled limit remains one equally weighted check.
Settling and overshoot retain their native full-state definitions, and effort
is mean squared applied input across all input channels, including trim.
It is not RMS, peak input, or input deviation from trim.

A finite fallback settling duration cannot pass if `settled` is false.
Undefined overshoot (e.g. no eligible constant-target state) is unavailable,
not automatically a zero overshoot. Missing/invalid data for ANY enabled check
leaves Rate 2 null rather than dividing by only the available checks.
Omitted/null limits disable checks; no enabled checks gives null.
Invalid limits or unsupported target names raise errors before report writing.
Reported numerical divergence, bound violations or verification errors fail
all configured checks and give Rate 2 zero.

Finalization saves `success_rate_1`, `success_rate_2` and
`success_rate_2_details` in each existing JSON. Details include the exact limits,
measured values, pass/fail/unavailable flags and reasons. The old `success_rate`
JSON key is replaced by `success_rate_1`, avoiding duplicate values. The
dashboard accepts that old key only as a Rate 1 fallback for historical files;
it never invents Rate 2 for unfinalized history.

Both figures independently average each run's score over the selected seeds
for the same LLM/artifact/test. Valid zeros count; null scores do not.
Rate 2 comparisons reject conflicting saved target definitions across seeds/
models: finalize all selected reports again after changing limits.
The core, tuning behavior and trajectory archives are unchanged.

## Dashboard

Configure `USE_MOCK_RESULTS` and `MODEL_FOLDERS_TO_PLOT` in
`plot_dashboard.py`. Mock mode reads only mock; real mode reads available
finalized reports for the selected models.

The success chart has one group of LLM bars per dynamic-model/test case,
with the plant name below that group. Cases are ordered by complexity:
cases sharing a level have a small gap, and different levels have a larger
gap with a separate complexity label beneath each level. Different plants
are never averaged together just because they have the same complexity.
Only repeated runs of the same plant/test/LLM are averaged in a success bar.
Adjust `CASE_GAP` and `COMPLEXITY_GAP` to change spacing.
Dense plant labels are wrapped and angled (`PLANT_LABEL_ROTATION`); the
complexity-label tier is placed below their rendered text to avoid overlap.
The other panels retain one bar per LLM for mean tokens/cost.
Missing values display as N/A, not zero.
Dense success labels are automatically hidden to prevent overlap. Click a
success bar in the interactive window to inspect the plant, IC, full-state
target, and RMSE of its contributing runs.

After changing artifact complexities, run `finalize_results.py` for all the
selected models before plotting: the dashboard uses the saved JSON rankings.
Conflicting rankings for the same case across LLMs raise an error instead of
silently drawing the case in different groups.

Two output figures:

- `plots/dashboard/mpc_dashboard_success_rate_1.png`
- `plots/dashboard/mpc_dashboard_success_rate_2.png`

Both keep the same token/cost panels; their success-vs-complexity panels use
different scores. To save without opening windows:

```powershell
python benchmarks/MPC/plot_dashboard.py --no-show
```

## State and input plots

Run `plot_trajectories.py` after the benchmark. No tuning, simulation, or API
calls are made; finalization is not required for these plots.

```powershell
python benchmarks/MPC/plot_trajectories.py --mock
python benchmarks/MPC/plot_trajectories.py --models groq-gpt-oss-120b --artifacts AUV.json
```

Alternatively, set `USE_MOCK_RESULTS`, `MODEL_FOLDERS_TO_PLOT`, `ARTIFACT_FILES`,
and `TEST_IDS` near the top of the script. `None` selects all saved artifacts
or tests. Mock mode ignores the real-model list. Figures are saved without
opening windows by default; use `--show` or set `SHOW_PLOTS = True` to view them.

Each test produces two files beneath
`plots/<LLM folder>/seed<value>/<artifact stem>/` (legacy unseeded reports
omit the seed folder):

- `<test_id>_states_vs_time.png`: one panel per state, with actual values and
  its dashed target/reference from the saved simulation. Time-varying references
  are preserved, not replaced with constant artifact targets.
- `<test_id>_inputs_vs_time.png`: one panel per applied input. Stair steps show
  each input over its actual control interval, including any trim already
  present in the saved applied input.

All saved samples are used. State/input names come from matching artifact
metadata when available; otherwise numbered labels are used. Defined
`state_bounds` and `input_bounds` appear as labeled red dash-dot horizontal
lines on their respective panels. Missing/null bounds draw no lines. These
lines show the matching **current artifact's** limits, not necessarily those
used by a historical run; the figure title marks this distinction. Bounds also
contribute to the displayed axis range so neither limit is hidden.
The script reads
linked NPZ archives and also accepts legacy inline verification arrays.
Missing or invalid trajectories are reported and skipped; they are never
silently regenerated. Existing plot files for the same test are replaced.

MPC uses one per-artifact JSON updated in place; Adaptive retains its
full/`_brief` format. Their code organization is aligned, but controller
algorithms, tuning parameters, and native metrics remain different. The reported
scores can differ in both normalization and aggregation.
