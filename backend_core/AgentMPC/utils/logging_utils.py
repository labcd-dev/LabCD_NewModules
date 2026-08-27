"""
================================================================================
utils/logging_utils.py
================================================================================
The original notebook used ~200+ scattered `print(...)` calls for everything
from "[MPC] Initialized..." to Actor/Critic reasoning dumps. That makes it
impossible to (a) turn verbosity down for a long tuning run, (b) persist a
run's log to disk for later inspection, or (c) tell info from warnings from
errors at a glance.

This module centralizes it behind the standard `logging` module. Call
``get_logger(__name__)`` from any module instead of using `print`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Call once, e.g. at the start of a run/notebook cell."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
