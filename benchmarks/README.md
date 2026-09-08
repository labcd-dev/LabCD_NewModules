# LabCD benchmarks

This directory contains the reproducible benchmark workflow for evaluating
LabCD controller-tuning agents across multiple dynamic plants, operating
conditions, scenario levels, and LLMs.

## Directory map

```text
benchmarks/
|-- artifacts/                 Plant definitions and generated test matrices
|-- dynamic_models/            Standalone dynamics examples used during development
|-- make_tests.py              Builds IC/target and scenario combinations
|-- score_complexity.py        Calculates every artifact complexity score
|-- ARTIFACTS.md               Artifact schema and scoring definitions
|-- MPC/                       MPC benchmark implementation and reports
`-- Adaptive/                  Reserved for the Adaptive-controller benchmark
```

## Artifacts

Each JSON file in `artifacts/` is a self-contained benchmark definition. It
contains:

- the plant name;
- the complete dynamic-model Python source in `python_code`;
- model metadata such as states, inputs, parameters, bounds, and derivative
  pairs;
- three initial-condition/target cases;
- three scenario definitions;
- the generated test combinations and their complexity values.

See [ARTIFACTS.md](ARTIFACTS.md) for the full schema and formulas.

## MPC benchmark

The MPC workflow is:

```text
Artifact definitions
        |
        v
make_tests.py            Create every IC/target x scenario test
        |
        v
score_complexity.py      Fill plant, condition, scenario, and total complexity
        |
        v
MPC/mpc_becnchmark.py    Tune MPC and verify the best parameters for each test
        |
        v
MPC/finalize_results.py  Add complexity, success rate, and refreshed LLM cost
        |
        v
MPC/plot_dashboard_revised.py
                         Plot performance, mean tokens, and mean cost
```

Run the complete pipeline from the repository root:

```powershell
.venv\Scripts\python.exe benchmarks\make_tests.py
.venv\Scripts\python.exe benchmarks\score_complexity.py
.venv\Scripts\python.exe benchmarks\MPC\mpc_becnchmark.py
.venv\Scripts\python.exe benchmarks\MPC\finalize_results.py
.venv\Scripts\python.exe benchmarks\MPC\plot_dashboard_revised.py
```

### 1. Generate tests

`make_tests.py` takes every declared initial-condition/target case and combines
it with every scenario. The current artifacts have three of each, producing
nine tests per plant with IDs such as `ic1_s1` and `ic3_s3`.

At this stage the script deliberately writes empty complexity placeholders:

- `tests.weights` values are `null`;
- every `tests.combinations[].total_cmplx` is `null`.

This keeps test generation separate from the scoring definition. Running the
script again rebuilds the matrix. Use `--keep-complexity` only when existing
scores should be retained.

### 2. Score complexity

`score_complexity.py` reads only the JSON metadata; it does not execute the
dynamics. It fills:

- `plant_cmplx` from the number of states;
- each IC/target `cmplx` from normalized position and rate errors;
- each scenario `cmplx` from its level;
- `tests.weights` with `0.30`, `0.30`, and `0.40`;
- every test's weighted `total_cmplx`.

Scores are stored both on `[0, 1]` and on the dashboard scale `[1, 10]`.

### 3. Run the MPC benchmark

`MPC/mpc_becnchmark.py` loads the embedded `python_code`, creates the dynamics,
applies the selected artifact scenario, runs the Actor/Critic/Terminator/Juror
tuning graph, and verifies the best MPC parameters in a final closed-loop
simulation.

The script supports two modes:

- `USE_MOCK_AGENTS = True`: fast, reproducible mock-agent testing with no API
  calls;
- `USE_MOCK_AGENTS = False`: runs every entry in `LLM_CONFIGS` using keys from
  the repository `.env` file.

`BENCHMARK_SELECTION` controls exactly which artifact and `test_id` values are
run. This is important when testing paid APIs. The benchmark checkpoints a
compact report after every run at:

```text
MPC/json_reports/results/<model-folder>/batch_tuning_results.json
```

The base report deliberately leaves `total_complexity` and `success_rate` as
`null`. It contains the test identity, compact tuning result, token usage, and
compact verification result without long trajectories.

### 4. Finalize results

`MPC/finalize_results.py` matches each result to its artifact using the dynamic
class name and `test_id`. It then:

- copies the matching artifact `total_cmplx` into `total_complexity`;
- computes the continuous success rate as `exp(-verification_rmse)`;
- refreshes real-LLM cost using the shared model-pricing resolver;
- writes `batch_tuning_results_full.json` without modifying the base report.

The boolean `verification.status.success` and the continuous `success_rate` are
different values. The former is the evaluator's pass/fail result; the latter is
the RMSE-based score used by the dashboard.

### 5. Plot the dashboard

`MPC/plot_dashboard_revised.py` reads one or more `_full.json` reports and
creates:

1. success score versus total complexity, one marker per test;
2. mean total tokens per LLM model;
3. mean estimated cost per LLM model.

Its default model folders come from `USE_MOCK_RESULTS` and
`DASHBOARD_LLM_MODEL_FOLDERS` in `MPC/plot_dashboard.py`. Full report paths can
also be supplied explicitly:

```powershell
.venv\Scripts\python.exe benchmarks\MPC\plot_dashboard_revised.py `
  benchmarks\MPC\json_reports\results\gpt-4o-mini\batch_tuning_results_full.json `
  benchmarks\MPC\json_reports\results\groq-gpt-oss-120b\batch_tuning_results_full.json
```

The image is written to `MPC/plots/mpc_dashboard_revised.png`.

For MPC-specific configuration and report details, see
[MPC/README.md](MPC/README.md).
