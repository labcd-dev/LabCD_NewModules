# Benchmark artifacts

`benchmarks/artifacts/` is the benchmark input library. Each JSON artifact is a
self-contained description of one dynamic plant and the tests that should be
run on it.

## Available plants

| Artifact | Dynamics class | States | Inputs |
|---|---|---:|---:|
| `cart_pole.json` | `CartPolePendulum` | 4 | 1 |
| `double_integrator.json` | `DoubleIntegratorDynamics` | 2 | 1 |
| `mass_spring_damper.json` | `MassSpringDamperDynamics` | 2 | 1 |
| `planar_drone.json` | `PlanarDroneDynamics` | 6 | 2 |
| `triple_mass_spring.json` | `TripleMassSpringDynamics` | 6 | 1 |
| `two_mass_spring.json` | `TwoMassSpringDynamics` | 4 | 2 |

With three IC/target cases and three scenarios, the full library currently
contains `6 x 3 x 3 = 54` tests.

## Artifact contents

```jsonc
{
  "system_name": "DoubleIntegratorDynamics",
  "python_code": "...complete executable dynamics source...",
  "metadata": {
    "n_states": 2,
    "n_inputs": 1,
    "state_names": ["position", "velocity"],
    "input_names": ["acceleration"],
    "params": {},
    "default_initial_state": [0.0, 0.0],
    "default_target": [0.0, 0.0],
    "state_bounds": {
      "lower": [-12.0, -8.0],
      "upper": [12.0, 8.0]
    },
    "input_bounds": {
      "lower": [-10.0],
      "upper": [10.0]
    },
    "derivative_pairs": [[0, 1]]
  },
  "plant_cmplx": {
    "normalized_score": 0.0,
    "score_1_to_10": 1.0
  },
  "Init Condition & Targets": [
    {
      "id": 1,
      "initial_state": [0.5, 0.0],
      "target_state": [0.0, 0.0],
      "cmplx": {}
    }
  ],
  "Scenarios": [
    {
      "level": 1,
      "name": "Nominal",
      "settings": {},
      "cmplx": {}
    }
  ],
  "tests": {
    "weights": {
      "plant_weight": 0.30,
      "ic_target_weight": 0.30,
      "scenario_weight": 0.40
    },
    "combinations": [
      {
        "test_id": "ic1_s1",
        "ic_target_id": 1,
        "initial_condition": [0.5, 0.0],
        "target": [0.0, 0.0],
        "scenario_level": 1,
        "scenario_name": "Nominal",
        "scenario_settings": {},
        "total_cmplx": {}
      }
    ]
  }
}
```

The important sections are:

- `system_name`: must match the class defined by `python_code`;
- `python_code`: complete plant code loaded by `DynamicLoader` during a run;
- `metadata`: dimensions, names, physical parameters, defaults, bounds, and
  position/rate index pairs;
- `Init Condition & Targets`: three complete state-vector pairs;
- `Scenarios`: nominal, noisy, and robust settings;
- `tests`: generated combinations and final total complexity.

## Build and score the artifacts

Run from the repository root:

```powershell
.venv\Scripts\python.exe benchmarks\make_tests.py
.venv\Scripts\python.exe benchmarks\score_complexity.py
```

`make_tests.py` creates every IC/target x scenario combination. By default it
sets the three weight values and every `total_cmplx` to `null`; this is the
expected intermediate state.

`score_complexity.py` then calculates and stores all component and total
scores. It validates vector shapes, finite bounds, derivative pairs, scenario
levels, and that the total weights sum to one. It never executes `python_code`.

## Complexity definition

All calculations first produce a normalized score in `[0, 1]`. The helper

```text
clip01(z) = min(1, max(0, z))
```

keeps every component within this range.

### 1. Plant complexity

The current version intentionally uses only the number of states:

```text
C_plant = clip01((n_states - 2) / (12 - 2))
```

The references mean:

- 2 states or fewer maps to `0`;
- 12 states or more maps to `1`;
- values between them are linearly interpolated.

Plant nonlinearity, open-loop stability, and number of inputs are not yet part
of this version of the score.

### 2. IC/target condition complexity

`metadata.derivative_pairs` identifies every `(position_index, rate_index)`.
For pair `j`, the position normalization reference is the farthest permitted
bound from the target:

```text
D_pos,j  = max(|lower_pos,j - target_pos,j|,
               |upper_pos,j - target_pos,j|)

D_rate,j = max(|lower_rate,j - target_rate,j|,
               |upper_rate,j - target_rate,j|)
```

The normalized initial errors are:

```text
e_pos,j  = clip01(|initial_pos,j  - target_pos,j|  / D_pos,j)
e_rate,j = clip01(|initial_rate,j - target_rate,j| / D_rate,j)
```

Errors across all derivative pairs are combined with root mean square:

```text
RMS(z) = sqrt(mean(z_j^2))

C_condition = 0.50 * RMS(e_pos) + 0.50 * RMS(e_rate)
```

Position distance and initial rate therefore contribute equally. RMS prevents
one difficult coordinate from disappearing among several easy coordinates.
The state bounds are the normalization reference, so the result can be
compared across differently scaled plants.

### 3. Scenario complexity

Scenario levels use the declared range 1 through 3:

```text
C_scenario = clip01((level - 1) / (3 - 1))
```

Therefore:

| Level | Name | Normalized complexity |
|---:|---|---:|
| 1 | Nominal | 0.0 |
| 2 | Noise | 0.5 |
| 3 | Robust | 1.0 |

The scenario settings stored in the artifact determine the actual measurement
noise, state masks, robust push, and optional physical-parameter perturbation
used by the benchmark. The numeric scenario-complexity component itself is
currently based only on the level.

### 4. Total complexity and weights

The three components are blended as:

```text
C_total = 0.30 * C_plant
        + 0.30 * C_condition
        + 0.40 * C_scenario
```

| Component | Weight | Meaning |
|---|---:|---|
| Plant | 0.30 | Size of the state-space model |
| IC/target condition | 0.30 | Initial position/rate distance from the target |
| Scenario | 0.40 | Nominal, noisy, or robust operating condition |

Scenario has the largest weight because environmental uncertainty and model
perturbation are intended to be the strongest difficulty change in this
benchmark version.

Every component and total is stored in two forms:

```text
normalized_score in [0, 1]
score_1_to_10 = 1 + 9 * normalized_score
```

The dashboard uses `score_1_to_10`.

## Success-rate definition

Success rate is calculated after the benchmark's best parameters have been
verified in closed loop. `finalize_results.py` uses:

```text
success_rate = exp(-verification_rmse)
```

This is a continuous score in `(0, 1]`, displayed as a percentage:

| Verification RMSE | Success rate |
|---:|---:|
| 0.0 | 100.00% |
| 0.5 | 60.65% |
| 1.0 | 36.79% |
| 2.0 | 13.53% |

Lower RMSE gives a higher score. A perfect zero-RMSE run gives `1.0`; the score
approaches zero as RMSE grows. If verification fails or has no RMSE, the result
remains `null`.

The formula currently uses the verification RMSE exactly as reported; it does
not normalize RMSE by each plant's state bounds. Cross-plant interpretation
therefore assumes that the chosen state scales and RMSE aggregation are
meaningful for the benchmark.

This continuous score is not the same as `verification.status.success`, which
is the evaluator's boolean success/failure decision based on its settling and
stability logic.

## Scenario behavior

The benchmark passes each combination's `scenario_settings` to
`apply_scenario_level`:

- level 1 is nominal;
- level 2 enables the artifact's configured noise;
- level 3 enables the robust configuration and may perturb physical
  parameters.

The declared IC and target remain the identity of the test. If a scenario
changes the state actually simulated, the result additionally records
`simulated_initial_condition` or `simulated_target`.

## Useful commands

| Command | Effect |
|---|---|
| `make_tests.py --dry-run` | Validate and report combinations without writing |
| `make_tests.py --keep-complexity` | Rebuild combinations while retaining matching scores |
| `score_complexity.py --dry-run` | Calculate and report scores without writing |
| `<script> artifacts\name.json` | Process only explicitly named artifacts |
| `<script> --artifacts-dir <dir>` | Use a different artifact directory |
