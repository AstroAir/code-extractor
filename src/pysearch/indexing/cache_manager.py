"""
Backward compatibility module for cache_manager.

This module provides backward compatibility for the old cache_manager module.
The functionality has been moved to the cache package.

DEPRECATED: Use pysearch.indexing.cache instead.
"""

from __future__ import annotations

import warnings

# Import everything from the new cache package
from .cache import (
    CacheBackend,
    CacheEntry,
    CacheManager,
    CacheStats,
    DiskCache,
    MemoryCache,
)

# Issue deprecation warning
warnings.warn(
    "Direct import from cache_manager is deprecated. Use pysearch.indexing.cache instead.",
    DeprecationWarning,
    stacklevel=2
)
