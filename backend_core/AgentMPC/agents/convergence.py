"""
================================================================================
agents/convergence.py
================================================================================
Tiny shared helper: has the tuning run's MSE history stopped improving
meaningfully? Used by:
  * agents/critic.py -- to force a strategy of "aggressive_explore" (bold,
    large parameter jumps) when normal explore/exploit has stalled, instead
    of waiting for the Terminator to eventually escalate to the Juror.
  * agents/terminator.py -- to help decide whether to escalate to the Juror.

Kept dependency-free (just numpy-free list math) so it can be imported from
either module without pulling in anything else.
"""

from __future__ import annotations

from typing import List


def is_plateaued(mse_history: List[float], window: int = 5, rel_tol: float = 0.01) -> bool:
    """True if the best MSE seen in the last `window` iterations hasn't
    improved by more than `rel_tol` (relative) over the best MSE seen
    before that window. False if there isn't enough history yet."""
    if len(mse_history) < window + 1:
        return False
    recent = mse_history[-window:]
    best_before = min(mse_history[:-window])
    return (best_before - min(recent)) / max(best_before, 1e-9) < rel_tol
