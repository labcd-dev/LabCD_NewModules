# MPC benchmark

This folder benchmarks the AgentMPC tuning workflow against the self-contained
plant tests in `benchmarks/artifacts/`. Each test tunes MPC parameters with mock
agents or a real LLM, verifies the best controller in closed loop, and stores a
compact JSON result.

## Run order

From the repository root:

```powershell
.venv\Scripts\python.exe benchmarks\make_tests.py
.venv\Scripts\python.exe benchmarks\score_complexity.py
.venv\Scripts\python.exe benchmarks\MPC\mpc_becnchmark.py
.venv\Scripts\python.exe benchmarks\MPC\finalize_results.py
.venv\Scripts\python.exe benchmarks\MPC\plot_dashboard_revised.py
```

The stages are:

1. `make_tests.py` creates every IC/target x scenario combination with empty
   complexity placeholders.
2. `score_complexity.py` fills the artifact complexity values.
3. `mpc_becnchmark.py` tunes and verifies MPC for the selected tests.
4. `finalize_results.py` adds total complexity, `exp(-RMSE)` success rate, and
   refreshed model cost to a separate `_full.json` report.
5. `plot_dashboard_revised.py` creates the customer-facing dashboard.

Detailed artifact schemas and formulas are documented in
[../ARTIFACTS.md](../ARTIFACTS.md).

## Select agents and tests

Edit the user-input section of `mpc_becnchmark.py`:

```python
USE_MOCK_AGENTS = True

BENCHMARK_SELECTION = {
    "double_integrator.json": ["ic1_s1", "ic2_s2"],
    "cart_pole.json": ["ic1_s1"],
}
```

- Mock mode ignores `LLM_CONFIGS` and makes no API calls.
- Real mode runs every dictionary in `LLM_CONFIGS` and reads its API key from
  `.env`.
- An empty test list selects all tests for that artifact.
- `BENCHMARK_SELECTION = None` selects every artifact and every test.

Use a small explicit selection before launching a paid real-LLM run.

## Results

Each agent/model has its own folder:

```text
json_reports/results/
|-- mock/
|   |-- batch_tuning_results.json
|   `-- batch_tuning_results_full.json
`-- <llm-model>/
    |-- batch_tuning_results.json
    `-- batch_tuning_results_full.json
```

The base report contains:

- plant name, state/input counts, test ID, IC, target, and scenario;
- best `Np`, `Nc`, `Q`, `R`, `P`, and `dt`;
- best cost/MSE, iterations, termination reason, and Juror verdict;
- token usage and estimated cost;
- compact verification metrics, status, solve time, and solver diagnostics;
- `null` placeholders for total complexity and success rate.

Long state, input, reference, and time trajectories are intentionally omitted.
The `_full.json` report preserves this compact structure and fills the two
placeholders. The base report is not overwritten.

Mock token counts and costs are reproducible simulated values for testing the
report pipeline. Real-LLM costs are calculated from prompt/completion usage and
the shared pricing table. If a returned model name has no known price, it is
listed under `unpriced_models` rather than silently assigned a price.

## Dashboard

`plot_dashboard_revised.py` shows:

- success score versus complexity for every test;
- mean token usage by LLM model;
- mean estimated cost by LLM model.

Choose default model folders with `USE_MOCK_RESULTS` and
`DASHBOARD_LLM_MODEL_FOLDERS` in `plot_dashboard.py`, or pass `_full.json` files
directly on the command line. The generated image is:

```text
plots/mpc_dashboard_revised.png
```
