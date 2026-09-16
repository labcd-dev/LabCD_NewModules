# Simple MPC benchmark

Each plant artifact contains its dynamics code, metadata, **one test**, and
one manually editable `total_complexity`. No test matrices, weights, separate
plant/condition scores, or automated complexity calculations are used.

## Files

- `artifacts/`: six current plant definitions with one editable test each.
- `dynamic_maker.py`: extracts each artifact's Python code into
  `artifacts/dynamic_models/<artifact-name>.py` for inspection or reuse.
- `mpc_benchmark.py`: tunes and verifies those tests, then saves compact JSON.
- `finalize_results.py`: copies your manual complexity and calculates success
  into a separate `_full.json`; preserves the base file.
- `plot_dashboard.py`: grouped success bars at discrete complexity levels, mean total
  tokens per LLM, and mean cost per LLM.

## Artifact format

```json
{
  "system_name": "DoubleIntegratorDynamics",
  "python_code": "... existing complete Python dynamics code ...",
  "metadata": { "n_states": 2, "n_inputs": 1 },
  "total_complexity": null,
  "test": {
    "initial_condition": [0.5, 0.0],
    "target": [0.0, 0.0],
    "scenario_level": 1,
    "scenario_name": "Nominal",
    "scenario_settings": {
      "noise_std_value": 0.0,
      "noise_state_mask": [false, false],
      "robust_push_scale": 1.0,
      "robust_state_mask": [false, false],
      "robust_noise_fraction": 0.0,
      "perturb_physical_params": false
    }
  }
}
```

The actual artifacts retain their full existing metadata and embedded Python
code. Fill the root `total_complexity` with an **integer 1–10**; the values
initially remain `null`. The same rating is used for this plant/test across
all LLM models. Use the same rating rules consistently for comparisons.
The IC and target arrays must contain one entry per state, including rates.

Scenario levels remain 1 (Nominal), 2 (Noise), and 3 (Robust). Adjust the name
and settings together when changing the level. Level 3 may change the actual
initial state and physical parameters; those changes are included in results.

## Configure the runner

To extract readable Python dynamics files, run from the repository root:

```powershell
python benchmarks/MPC/dynamic_maker.py
```

The sources are copied verbatim and syntax-checked, not executed. Rerunning
replaces matching exported `.py` files; the artifact JSON is never modified.
The benchmark still reads the JSON `python_code` as its source of truth:
editing an exported `.py` file does not change the benchmark automatically.

Edit the settings near the top of `mpc_benchmark.py`:

- `ARTIFACT_FILES = None` selects all six plants; set a filename list for a subset.
- `USE_MOCK_AGENTS = True` makes no API calls but still uses the real simulator.
- With mock mode off, `CHEAPEST_MODEL_ONLY = True` runs only the cheapest
  configured entry using the repository's price table and an equal input/output
  token estimate. Actual costs use recorded input/output counts. With the
  current configured rates, GPT-4.1 Nano is selected.
- Set `CHEAPEST_MODEL_ONLY = False` to run every model in `LLM_CONFIGS`.
- To force Groq, leave only the `groq-gpt-oss-120b` configuration in that list.
- `MAX_ITERATIONS`, timing, guidance, cost weights, and optional seed MPC
  parameters are editable in the same settings section.

Real-model mode uses the repository `.env` (`OPENAI_API_KEY` or `GROQ_API_KEY`).
No credentials are stored in the artifacts or result files.

## Run

From the repository root, after activating the existing environment:

```powershell
python benchmarks/MPC/mpc_benchmark.py
python benchmarks/MPC/finalize_results.py
python benchmarks/MPC/plot_dashboard.py
```

The first command performs tuning and can use paid APIs. The other two never
call an LLM or run a simulation. For saving a plot without opening a window:

```powershell
python benchmarks/MPC/plot_dashboard.py --no-show
```

In `finalize_results.py`, set `USE_MOCK_RESULTS = True` to update only `mock`;
the model list and explicit CLI input paths are ignored in mock mode. With it
set to `False`, `MODEL_FOLDERS_TO_UPDATE` chooses the model folders to update.
Missing selected reports are announced and skipped, without falling back to
other models. In non-mock mode, explicit CLI input paths override the list.

## Reports and success score

Reports are stored under `json_reports/results/<LLM-folder>/` (or `mock/`):

- `batch_tuning_results.json`: plant dimensions, IC, target, scenario,
  best MPC parameters, cost, MSE, iterations, termination, verdict, token usage,
  verification metrics/status/timing/solver diagnostics, and two null placeholders.
- `batch_tuning_results_full.json`: the same compact report, with the artifact's
  manual complexity and success score filled in.

No timestep states, inputs, references, times, or agent histories are saved.
Failed runs remain identifiable by their error and have a null success score
if verification RMSE is unavailable.
Each benchmark invocation replaces the selected model's standard base report;
copy that report elsewhere first if you want to keep multiple independent runs.

The success score is normalized by the artifact's state bounds:

`success_rate = exp(-verification_rmse / bound_scale)` (0–1; dashboard displays percent).

`bound_scale = sqrt(mean((upper - lower)^2))`, using all state dimensions in
the artifact's `metadata.state_bounds`. The finalized report records this as
`success_rate_bound_scale`. Bounds must be finite, correctly sized, and have
positive ranges. Zero RMSE gives success 1; RMSE equal to the scale gives
approximately 0.367879. Initial-error normalization is not used.

Keep bounds fixed when comparing LLMs: wider bounds give higher scores for
the same RMSE. This scalar normalization still combines states with potentially
different physical units; it is a benchmark score, not a probability of success.

Change only `total_complexity`, then rerun finalization and plotting to update
the x positions without paying for another tuning run. If IC/target/scenario
changes, rerun that plant's benchmark; the finalizer rejects stale conditions.
Changing dynamics code or scenario settings likewise requires fresh results.
Manual complexities still set to `null` cannot appear in the complexity panel.

## Dashboard

Edit `USE_MOCK_RESULTS` and `MODEL_FOLDERS_TO_PLOT` in the plotting file.
Missing selected model folders are reported and skipped; at least one finalized
report is needed. The first panel has one group for each assigned integer
complexity level, with a side-by-side bar for each LLM model. Multiple verified
runs at the same level are averaged within that model. Missing verification
scores are shown as `N/A`, not zero; a real zero success score remains zero.

- `SHOW_SUCCESS_VALUES = True`: show mean success percentages on the bars.
- The success panel is wider; narrow bars use vertical value labels, with
  high-success labels placed inside bars to avoid crowding near 100%.
- Colors consistently identify the same LLM across all three panels.
- Clicking a success bar shows the LLM, complexity, number of included runs,
  mean success, and individual plant/IC/target/RMSE details (up to three runs).
- There is no smoothing or raw-point plotting in this bar-chart version.

The full old workflow remains in `benchmarks_backup_2026-09-16/MPC/` from the
repository root (or `../../benchmarks_backup_2026-09-16/` from this folder).
