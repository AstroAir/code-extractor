"""
Backward compatibility semantic module.

This module provides backward compatibility by re-exporting functions
from the search.semantic module.

DEPRECATED: Import directly from pysearch.search.semantic instead.
"""

from __future__ import annotations

import warnings

# Import all functions from the main location
from .search.semantic import *  # noqa: F403, F401

# Issue deprecation warning
warnings.warn(
    "Direct import from pysearch.semantic is deprecated. "
    "Use 'from pysearch.search.semantic import <function>' instead.",
    DeprecationWarning,
    stacklevel=2,
)
