"""
================================================================================
dynamics/trajectory_loader.py
================================================================================
Loads and validates a user-supplied "trajectory plugin" (.py file) -- the
reference-trajectory equivalent of dynamics/loader.py's DynamicLoader.

Contract (see agents/trajectory_validator.py:TRAJECTORY_STANDARD for the full
human-readable version):

    def create_trajectory(dt_mpc: float, simulation_time: float,
                           n_states: int, state_names: list[str]) -> np.ndarray:
        '''Return shape (n_steps, n_states), n_steps >= simulation_time/dt_mpc.'''
        ...

Like DynamicLoader, this is the ONLY place that reflects on / probes a
trajectory file's shape -- everywhere else can trust the contract once a
TrajectoryLoader has been constructed successfully.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np


class TrajectoryPluginError(Exception):
    """Raised whenever a trajectory file does not satisfy the plugin contract."""


@dataclass
class TrajectoryLoader:
    generate: Callable[[float, float, int, List[str]], np.ndarray]
    source_name: str

    @classmethod
    def load_from_path(cls, path: str) -> "TrajectoryLoader":
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise TrajectoryPluginError(f"File not found: {source_path}")
        if source_path.suffix != ".py":
            raise TrajectoryPluginError(f"Expected a .py file, got: {source_path.suffix}")

        module_name = f"agent_mpc_trajectory_{source_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise TrajectoryPluginError(f"Could not load module spec for {source_path}")

        module = importlib.util.module_from_spec(spec)
        module.__dict__["np"] = np  # same auto-injection convention as DynamicLoader

        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as e:  # noqa: BLE001
            raise TrajectoryPluginError(f"Error while executing '{source_path.name}': {e}") from e

        if not hasattr(module, "create_trajectory"):
            raise TrajectoryPluginError(
                f"'{source_path.name}' must define create_trajectory(dt_mpc, simulation_time, "
                f"n_states, state_names) -> np.ndarray"
            )

        fn = module.create_trajectory
        if not callable(fn) or len(inspect.signature(fn).parameters) < 4:
            raise TrajectoryPluginError(
                f"create_trajectory in '{source_path.name}' must accept exactly "
                f"(dt_mpc, simulation_time, n_states, state_names)"
            )

        cls._validate(fn, source_path.name)
        return cls(generate=fn, source_name=source_path.name)

    @staticmethod
    def _validate(fn: Callable, source_name: str) -> None:
        """Smoke-test the function with representative arguments (2 states,
        a short horizon) and check the output shape/finiteness -- exactly
        the same "call it once with sane defaults and check the contract"
        approach DynamicLoader uses for dynamics()."""
        dt_mpc, simulation_time, n_states = 0.02, 1.0, 2
        state_names = ["state_0", "state_1"]
        try:
            out = fn(dt_mpc, simulation_time, n_states, state_names)
        except Exception as e:  # noqa: BLE001
            raise TrajectoryPluginError(f"create_trajectory() raised when called: {e}") from e

        out = np.asarray(out)
        expected_min_steps = int(simulation_time / dt_mpc)
        if out.ndim != 2 or out.shape[1] != n_states:
            raise TrajectoryPluginError(
                f"create_trajectory() must return shape (n_steps, n_states) -- got {out.shape} "
                f"for n_states={n_states}"
            )
        if out.shape[0] < expected_min_steps:
            raise TrajectoryPluginError(
                f"create_trajectory() returned only {out.shape[0]} steps for simulation_time="
                f"{simulation_time}s at dt_mpc={dt_mpc}s (need at least {expected_min_steps})"
            )
        if not np.all(np.isfinite(out)):
            raise TrajectoryPluginError("create_trajectory() returned non-finite values (NaN/Inf).")
