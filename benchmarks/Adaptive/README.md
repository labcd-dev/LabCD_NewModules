# Adaptive benchmark

The Adaptive benchmark has four user-facing scripts:

1. `adaptive_benchmark.py` runs the selected artifacts with mock or real agents.
2. `finalize_results.py` updates that same accepted-format JSON with two scores and target checks.
3. `plot_dashboard.py` creates two figures, one for each score, with token/cost panels.
4. `plot_trajectories.py` plots all states and applied inputs against time,
   with target curves only for declared controlled outputs.

## Artifacts

The active `artifacts/` directory contains six models adapted from
`py files/jsons/`:

- `AUV.json`
- `CartPole.json`
- `DCMotor.json`
- `InvPendulum.json`
- `QuanserHelicopter.json`
- `TwoLinkManipulator.json`

Additional models are retained under `challenging_artifacts/` and are not
selected by the benchmark's active artifact scan:

- `non_affine_artifacts/BallBeam.json`
- `non_zero_trim_artifacts/AircraftPitch.json`
- `non_zero_trim_artifacts/CSTR.json`
- `non_square_artifacts/QuanserHeliCh1.json`
- `non_square_artifacts/QuanserHeliCh2.json`
- `non_square_artifacts/TwoLinkManipCh1.json`
- `non_square_artifacts/TwoLinkManipCh2.json`

Each active artifact owns its plant model, metadata, input and output lists,
simulation settings, tuning settings, and tests. The benchmark does not
duplicate artifact-specific simulation time, step size, target RMS fraction,
tuning rounds, or objective priorities.

The benchmark passes each artifact directly to the normal Adaptive plant
compiler and tuning pipeline. It does not add a trim/feed-forward input or
rewrite the artifact's input channels.

Before invoking an agent, it skips artifacts marked
`"control_input_affine": false` and MIMO artifacts whose input and output
counts are unequal. These structures are unsupported by the current Adaptive
backstepping/SMC workflow.

`fixed_targets` are preserved in the result's `tuning_inputs`. The current
Adaptive core does not accept separate MSE, settling-time, overshoot, and
control-effort limits as termination constraints, so these values are not
individually enforced by the tuner.
Finalization evaluates these limits offline for Success Rate 2; this does not
change the tuner's objectives, stopping criteria or controller gains.

## Running the workflow

From the repository root:

```powershell
python benchmarks/Adaptive/adaptive_benchmark.py
python benchmarks/Adaptive/finalize_results.py
python benchmarks/Adaptive/plot_dashboard.py --no-show
python benchmarks/Adaptive/plot_trajectories.py
```

The benchmark saves a separate accepted-format report for each artifact/LLM/seed:

```text
json_reports/results/<model>/seed<value>/<artifact stem>_results.json
# Example: json_reports/results/gpt-4.1-nano/seed42/AUV_results.json
# Mock:    json_reports/results/mock/seed42/AUV_results.json
```

Each file is an array of that artifact's test records, checkpointed after
each test. Rerunning an artifact replaces only its report; other artifacts,
models, seeds, and existing historical batch reports are left untouched. Filenames
contain no date. Writes are atomic, so an interrupted write does not truncate
the previous checkpoint.

Saved fields follow `json_reports/accepted_results_template.json`. This file
must exist: the benchmark reads its field structure before running and saving.
Its explanation strings are documentation, not literal saved values. The JSON
keeps the plant/test information, applied system specification, tuning inputs,
best parameters/metrics/round, agent messages, token usage, verification,
summary, abstract, and any failure details that are available.
`core_outputs`, tuning `history`, and `tuning_history` are not saved. Other
fields absent from the accepted template are also omitted. Plant/provider
keys represented by `<placeholder>` entries are preserved. Optional fields
are not invented, and the template's explanation strings are never used as
result values. The complete returns remain available internally while the
benchmark runs, without changing the core.

The final trajectory arrays are saved only once in a compressed file:

```text
json_reports/results/<model>/seed<value>/trajectories/<artifact>_<test_id>_verification.npz
```

`series.trajectory_file` links to this file relative to the results JSON.
`series.channels` retains labels, ordered column names, and each channel's
`array_key`. Its large `data` arrays are not embedded in the results JSON.
The NPZ stores `times`, `states` (all states), `outputs`
(controlled outputs), `reference` (output targets), and `inputs`. All rows
follow the exported timestamps. Nothing is further downsampled by the benchmark.
Nonfinite numbers become JSON `null`; in trajectory arrays missing/nonfinite
measurements are retained as `NaN`/infinity. Symbolic values remain strings.

For example, to load trajectories for plotting:

```python
import json
from pathlib import Path
import numpy as np

report_path = Path("benchmarks/Adaptive/json_reports/results/mock/seed42/AUV_results.json")
record = json.loads(report_path.read_text(encoding="utf-8"))[0]
with np.load(report_path.parent / record["series"]["trajectory_file"], allow_pickle=False) as data:
    times = data["times"]
    states = data["states"]
    targets = data["reference"]  # Output targets, not a full-state reference.
    inputs = data["inputs"]
```

This layout applies to new benchmark saves. Existing inline reports are not
automatically rewritten. Keep the JSON and its `trajectories/` folder together.

### Seeds and progress

Set `RUN_SEEDS = [42, 123, 2024]` in `adaptive_benchmark.py`. Every selected
LLM/artifact/test starts a fresh workflow for each seed. Real-agent sampling
and mock proposals/token counts use that seed. Normal agent role temperatures
and token limits are retained. Provider-supported seeds are best-effort, not
a guarantee of identical responses. Three seeds mean three times as many
runs and potentially API usage.

The core is unchanged. Its RBF-centre base seed stays 0 (backstepping stages
use offsets); its uncertainty-dependence probe also stays fixed. This keeps
controller-initialization randomness separate from LLM variability.
Simulation is deterministic for the currently defined disturbance expressions;
no additional simulation/noise seed is injected. Initial states, references,
uncertainty/disturbance expressions, objectives and simulation settings remain
the same across repetitions.

Each saved result record has `seeds`: `run`, `llm`, `mock`,
`estimator` and `simulation`. The last is null because there is no separate
simulation RNG here; mock-only/real-only seeds are null when inactive.
Real factory seeding is scoped to this sequential benchmark call and restored
afterwards, including on failure. No controller or core source files are edited.

`json_reports/results/benchmark_progress.json` lists completed, remaining,
running and failed combinations with a done/total count. Completion means the
pipeline finished and its files were saved, not that the controller met its
tracking target. Initial-controller-only runs also count as done when tuning
is disabled. Failed attempts stay in the remaining list; interruptions are
checkpointed. The tracker inventories matching saved seeded runs at startup,
but does not automatically skip/resume them. Starting the benchmark reruns
all selected combinations. Selection matching checks the plant/test/LLM/seed
and test conditions, not every historical tuning setting.

### Finalization and success rate

The finalizer selects `<artifact>_results.json` from each configured model
folder's selected seed subfolders (or only `mock` when `USE_MOCK_RESULTS=True`).
It updates `<artifact>_results.json` in place, using the accepted template.
No `_brief` or `_full` copies are created. Agent messages, best parameters,
applied system specs and trajectory metadata stay in the same file; there is
no separate compact-result structure with `dynamics` or `token_counts`.
If no per-artifact reports exist, legacy batch reports remain supported.
Trajectory archives are never rewritten by finalization. Real-run missing
cost estimates, when needed, are added to `tuning.token_usage`, not a duplicate
token-count section. `SEEDS_TO_UPDATE = None` selects all saved seeds;
`[42]` selects seed42. Once any seeded reports exist for a model, old unseeded
history is ignored instead of mixed into the new results. The progress JSON is
not a controller report and is ignored by finalization/plotting.

```powershell
python benchmarks/Adaptive/finalize_results.py --models gpt-4.1-nano --seeds 42 123 2024
python benchmarks/Adaptive/plot_dashboard.py --models gpt-4.1-nano --seeds 42 123 2024 --no-show
python benchmarks/Adaptive/plot_trajectories.py --models gpt-4.1-nano --seeds 42 123 2024
```

The score now matches the current MPC implementation:

```text
task_scale_mse[i] = core task_scale[i] ** 2
success_rate_1 = exp(-10 * mean_i(tracking_mse.full[i] / task_scale_mse[i]))
```

Here `i` identifies a declared controlled output, not an internal state or
tuning round. Each MSE is normalized separately before averaging the ratios.
The core's full-resolution tracking MSE and RMS task scale are used directly;
the finalizer does not recalculate Rate 1 from plotting data.
Verification retains the original RMS `task_scale` and adds
`task_scale_mse` and ordered `outputs`. Repeated scoring does not square the
MSE scales again. A zero scale with nonzero MSE scores zero. Failed finite or
bounded checks score zero; missing/invalid score data gives `null`. Missing
the steady-state tuning target alone does not force a zero score.

The calculators are independent modules:

- `success_rate/success_rate_1.py`: the existing exponential formula, unchanged.
- `success_rate/success_rate_2.py`: `passed_targets / enabled_targets`.

Rate 2 gives 75% when three out of four fixed targets pass. This is partial
target credit, not the paper's strict full-task pass rate. Every enabled
target must pass to get 100%.

| Artifact limit | Adaptive verification comparison |
| --- | --- |
| `mse_max` | Mean full-window MSE of the declared controlled outputs <= limit |
| `settling_time_max_s` | `settling_time` <= limit AND `settling_time_reached` true |
| `overshoot_max_pct` | Largest eligible controlled output's true target-crossing overshoot, in percent, <= limit |
| `control_effort_max` | Mean of squared per-input `control_rms` values <= limit |

Control effort is `mean_i(RMS(u_i)**2)`, not RMS itself or the square of the
mean RMS. This matches MPC's mean squared applied-input definition.
Settling retains Adaptive's native output/reference-band definition; it is
not recomputed using MPC's different full-state tolerance/hold rule.

Adaptive's original `overshoot_frac` is NOT used for Rate 2: it includes initial
tracking error and normalizes by reference amplitude. The calculator reads the
saved final output/reference arrays, checks output names, and detects excursions
past each constant target, relative to that output's actual initial gap:
`overshoot_pct_i = 100 * largest_error_on_opposite_side_of_target / abs(initial_error_i)`.
No target crossing gives 0%; moving references and initial gaps <= 1e-6 are
ineligible. The largest eligible controlled-output percentage is compared.
If none is eligible, the overshoot check is unavailable, not falsely passed.
Original core metrics, including `overshoot_frac`, stay unchanged.

An export marked `downsampled: false` permits full pass/fail checking.
Downsampled/unknown-coverage samples can prove failure if an observed peak
already exceeds the limit, but cannot prove a pass; a smaller peak therefore
leaves that check unavailable. Missing/invalid archives do the same. The
benchmark now uses the core's existing `LABCD_ADAPTIVE_SERIES_MAX_POINTS` knob
per run to retain every final sample for its artifact's horizon/step, then
restores the environment. This changes export size, not the core, simulations,
gains or tuning decisions. Historical downsampled archives are not reconstructed.

Omitted/null limits disable checks. Missing/undefined data for any enabled
check leaves Rate 2 null, rather than inflating its score by dropping a check.
No enabled targets also gives null. Failed numerical finite/bounded checks
give zero for all configured targets, regardless of placeholder MSE values.
The same result JSON stores `success_rate_1`, `success_rate_2` and
`success_rate_2_details` with limits, measurements, flags, sources and reasons.
The former single `success_rate` key is no longer written.
Original verification measurements and trajectory archives remain preserved.
Finalization validates every record before atomically replacing a result file.
The accepted template includes these scoring fields and must be present; it
does not supply literal result values or remove the existing accepted content.
Re-running the benchmark replaces its selected model/seed/artifact's report
and resets score placeholders; run finalization again before plotting.

Old `*_results_brief.json` files are not selected by either plotter or the
finalizer. The migration archives them outside active `results/` under
`json_reports/brief_results_backup_*/`, retaining their model/seed paths so
they can be recovered. The active layout is simply:

```text
json_reports/results/<model>/seed<value>/
    AUV_results.json
    CartPole_results.json
    ...
    trajectories/
```

### Trajectory plots

The plotter follows MPC's settings, CLI options, figure layout, and folder
structure. Set `USE_MOCK_RESULTS`, `MODEL_FOLDERS_TO_PLOT`, `ARTIFACT_FILES`,
and `TEST_IDS` in `plot_trajectories.py`, or use command-line options:

```powershell
python benchmarks/Adaptive/plot_trajectories.py --mock --artifacts AUV.json
python benchmarks/Adaptive/plot_trajectories.py --models groq-gpt-oss-120b --show
```

Saved images:

```text
plots/<model>/seed<value>/<artifact>/<test_id>_states_vs_time.png
plots/<model>/seed<value>/<artifact>/<test_id>_inputs_vs_time.png
plots/dashboard/adaptive_dashboard_success_rate_1.png
plots/dashboard/adaptive_dashboard_success_rate_2.png
```

All saved samples are plotted without another simulation or API call. The
plotter reads linked NPZ files and also supports old inline series. References
are mapped by saved output names; non-controlled states have no invented
targets. Dashboard plots use those same finalized per-artifact result files,
reading specs from `applied_system_spec.dynamics` and usage from
`tuning.token_usage`.

The dashboard uses MPC's three-panel layout: success bars per plant/test
(comparing models), mean token usage per model, and mean cost per model.
Different plants sharing a complexity are not averaged into one bar.
Each success bar is the arithmetic mean of available saved success rates
across selected seeds for that same LLM/artifact/test, independently for each
score: `mean_seed(success_rate_1)` or `mean_seed(success_rate_2)`.
It is not a score calculated from mean tracking errors. Null scores are
excluded, valid zeros are included, and clicking a bar lists contributing
seeds/run counts. In each plotter, `SEEDS_TO_PLOT = None` selects all saved
seeds; an integer list selects a subset. Their filters are independent.
Trajectory plots are never averaged: each seed gets separate figures and
loads its own linked archive relative to its result JSON.
Duplicate LLM/case/seed records, mismatched folder/record seeds, changed test
conditions or contradictory complexities raise errors rather than silently
biasing a mean. Historical unseeded-only models keep their old layout.
`CASE_GAP`, `COMPLEXITY_GAP`, and `PLANT_LABEL_ROTATION` match MPC's defaults;
names are wrapped/angled, with a separate complexity-label tier below them.
Clicking a success bar shows that case's initial state, references, and RMSE.

Active Adaptive test complexities match the corresponding MPC tests:
AUV/DC Motor = 1, Cart-Pole/Inverted Pendulum = 2,
Quanser Helicopter/Two-Link Manipulator = 3. These are shared manual labels,
not a guarantee of equal controller performance. After changing artifact
rankings, rerun `finalize_results.py` for the selected models to update their
existing reports before plotting; no retuning is needed if conditions are unchanged.

The trajectory file preserves all **exported** samples, not hidden controller
internals. The unchanged core defaults to a 2,000-point export cap, but the
benchmark now sets that existing cap high enough to retain its final samples.
The core does not return all trial
trajectories or every raw real-tuner reply. The benchmark does not reconstruct
or silently invent those missing data.

### Reading the full-output template

`json_reports/results_template.json` is an explanatory example of the full
result structure, with descriptions instead of actual values. It is not a
report, validation schema, or simulation input. It includes the union of
observed fields plus optional fields checked against the core (real message
metadata, tuning attempts, estimator diagnostics, and failure details).
`[Optional]`, method-specific labels, and placeholder keys describe branches
that need not occur in every run. Arrays show a representative element/row;
they do not limit the number of samples, channels, or attempts.

Both explanatory templates keep `series.channels.*.data` visible so you can
see the array structure. These are documentation examples only: actual new
reports store the arrays in the linked NPZ. The accepted template still omits
the history and `core_outputs` sections you removed and controls which
fields the benchmark saves. The full template remains a reference for the
complete internal returns and older unfiltered reports.

- `tuning.history`: round 0 is the initial design; later entries are tuner
  attempts. An unusable reply need not produce a new simulation.
- `tuning.best_metrics` / `core_outputs.best`: the selected candidate's tuning
  evaluation, not automatically the final or last simulation. Best selection
  uses the core's eligible objective score; it need not minimize every metric.
- `verification` / `core_outputs.result.final_metrics`: the final simulation;
  normally best gains, but the core can revert to proven initial gains if the
  best candidate's final rerun fails. With tuning disabled this can be the
  initial build, not a separate rerun.
- Tracking arrays follow declared **outputs**; control-effort arrays follow
  **inputs**; initial conditions and `series.channels.x` follow all **states**.
- `tracking_pct_headline` is the worst **steady** output percentage, not full
  accuracy. Tracking percentage uses linear clipping of normalized RMSE, not
  the exponential score. Finalization adds both scores to the same accepted
  result JSON; benchmark runs start with unscored placeholders.
- `task_scale` is RMS, not MSE. `overshoot_frac` is maximum absolute error
  divided by reference amplitude, so zero references can yield huge values.
- `fixed_targets` are offline Rate 2 limits, not hard tuning stopping constraints.
- In nonfinite failed runs, the core can use zero tracking-MSE placeholders;
  always inspect numerical health/success checks before interpreting them.
- Repeated sections are copies for convenience, not additional simulations.
  Real Tuner replies absent from the core return are not invented in this guide.

## Supporting files

The four scripts above are the files normally run by the user. Some additional
files are still required internally:

- `mock/` implements API-free mock agents.
- `fast_report.py` avoids expensive symbolic report simplification for the
  two-link models without changing their controller or simulation.
- `success_rate/` contains the two offline scoring calculators.

The accepted template controls the one saved result structure; finalization
extends that same record with scores without creating another report format.
Artifact-backup directories are not used during benchmark execution and are
retained only as recoverable backups.
