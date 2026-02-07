"""
Backward compatibility API module.

This module provides backward compatibility by re-exporting PySearch
from the core.api module.

DEPRECATED: Import directly from pysearch instead.
"""

from __future__ import annotations

import warnings

# Import PySearch from the main location
from .core.api import PySearch

# Issue deprecation warning
warnings.warn(
    "Direct import from pysearch.api is deprecated. Use 'from pysearch import PySearch' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
__all__ = ["PySearch"]
