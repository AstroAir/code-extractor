"""
Backward compatibility configuration module.

This module provides backward compatibility by re-exporting SearchConfig
from the core.config module.

DEPRECATED: Import directly from pysearch instead.
"""

from __future__ import annotations

import warnings

# Import SearchConfig from the main location
from .core.config import SearchConfig

# Issue deprecation warning
warnings.warn(
    "Direct import from pysearch.config is deprecated. Use 'from pysearch import SearchConfig' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
__all__ = ["SearchConfig"]
