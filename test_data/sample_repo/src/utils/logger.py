"""Logging configuration and utilities.

Provides a centralized logger factory with configurable output levels
and format options.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

_configured = False


def setup_logging(
    level: str = "INFO",
    stream: TextIO = sys.stderr,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """Configure the root logger for the application.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        stream: Output stream for log messages.
        fmt: Log message format string.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("sample_app")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured Logger instance.
    """
    setup_logging()
    return logging.getLogger(f"sample_app.{name}")
