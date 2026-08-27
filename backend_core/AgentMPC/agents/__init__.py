"""Agent nodes for the MPC tuning graph.

Note: importing this package's LLM-based modules (actor, critic, scenarist,
terminator, juror) requires `pydantic` and `langchain_core`. `evaluator.py`
and `numeric_tuner.py` do not -- they only depend on numpy and the mpc/
sub-package, so they can be used standalone (e.g. from a script or notebook)
without pulling in any LLM dependency.
"""
