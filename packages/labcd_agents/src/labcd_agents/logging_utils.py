"""Small logging convenience wrapper.

Several modules ship their own ``get_logger`` / ``log_to_file`` helpers that
all boil down to "give me a configured ``logging.Logger``". This module
provides one consistent implementation; modules that need custom handlers
(e.g. writing to a per-run log file) can still pass their own logger into
:class:`~labcd_agents.agent.BaseAgent`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

__all__ = ["get_logger"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The log level defaults to the ``LABCD_LOG_LEVEL`` environment variable,
    falling back to ``INFO``. Handlers are only attached once per logger name
    so repeated calls (e.g. one per agent instantiation) don't duplicate
    log lines.
    """
    logger = logging.getLogger(name)
    resolved_level = (level or os.getenv("LABCD_LOG_LEVEL") or "INFO").upper()
    logger.setLevel(resolved_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
