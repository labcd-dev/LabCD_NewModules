# Adaptive benchmark

The Adaptive benchmark is independent from the MPC benchmark while its format
is being developed and checked.

## Files

- `artifacts/` contains direct AgentAdaptive `system_spec` JSON objects.
- `make_tests.py` adds Adaptive-only bounds, IC/target cases, scenarios, and the
  nine test combinations for each artifact. Complexity fields start as `null`.
- `score_complexity.py` reads the symbolic Adaptive equations directly and
  fills plant, condition, scenario, and total complexity.

Run from the repository root:

```powershell
.venv\Scripts\python.exe benchmarks\Adaptive\make_tests.py
.venv\Scripts\python.exe benchmarks\Adaptive\score_complexity.py
```

The artifacts define the double integrator, mass-spring-damper,
two-mass-spring, triple-mass-spring, cart-pole, and planar drone. They do not
contain an MPC plugin or a shared AgentPlant wrapper.

The `benchmark` object is module-local configuration used by the Adaptive
benchmark scorer. It holds bounds, dimensions, derivative pairs, and the plant
complexity mark. It is not passed to the Adaptive controller; the controller
receives the top-level `status`, `system_name`, and `dynamics` fields.

## Complete tuning example

`artifacts/adaptive_tuning_test.py` is a standalone learning example. It defines
a numerical Python double-integrator plant, converts the matching AgentPlant
metadata to an Adaptive `system_spec`, configures all public tuning-run inputs,
uses Groq for the Design/Tuner/Reporter roles, and saves its result under
`Adaptive/results/`.

Add `GROQ_API_KEY` to the repository `.env`, then run:

```powershell
.venv\Scripts\python.exe benchmarks\Adaptive\artifacts\adaptive_tuning_test.py
```
