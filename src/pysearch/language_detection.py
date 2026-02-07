"""
Backward compatibility language_detection module.

This module provides backward compatibility by re-exporting functions
from the analysis.language_detection module.

DEPRECATED: Import directly from pysearch instead.
"""

from __future__ import annotations

import warnings

# Import all functions from the main location
from .analysis.language_detection import *  # noqa: F403, F401

# Issue deprecation warning
warnings.warn(
    "Direct import from pysearch.language_detection is deprecated. Use 'from pysearch import <function>' instead.",
    DeprecationWarning,
    stacklevel=2,
)
