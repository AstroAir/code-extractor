"""
Backward compatibility types module.

This module provides backward compatibility by re-exporting types
from the core.types module.

DEPRECATED: Import directly from pysearch instead.
"""

from __future__ import annotations

import warnings

# Import all types from the main location
from .core.types import *  # noqa: F403, F401

# Issue deprecation warning
warnings.warn(
    "Direct import from pysearch.types is deprecated. Use 'from pysearch import <type>' instead.",
    DeprecationWarning,
    stacklevel=2,
)
