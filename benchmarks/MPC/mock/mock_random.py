"""Small deterministic-random helpers shared by the mock MPC agents."""

from __future__ import annotations

import random
from typing import Any, Dict


def mock_rng(state: Dict[str, Any], node_offset: int) -> random.Random:
    """Return a reproducible RNG unique to a run, iteration, and node."""
    base_seed = int(state.get("mock_seed", 42))
    iteration = int(state.get("iteration", 0))
    return random.Random(base_seed + iteration * 1009 + node_offset)


def log_uniform(rng: random.Random, low_exponent: float, high_exponent: float) -> float:
    """Sample a positive value uniformly in base-10 log space."""
    return 10.0 ** rng.uniform(low_exponent, high_exponent)
