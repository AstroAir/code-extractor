"""
Cache package for pysearch.

This package provides comprehensive caching capabilities for search results
with multiple backend options and automatic invalidation.

Public API:
    CacheManager: Main cache management interface
    CacheEntry: Cache entry data structure
    CacheStats: Cache performance statistics
    CacheBackend: Abstract base for cache backends
    MemoryCache: In-memory cache implementation
    DiskCache: Persistent disk-based cache
"""

from .backends import CacheBackend, DiskCache, MemoryCache
from .manager import CacheManager
from .models import CacheEntry, CacheStats

__all__ = [
    "CacheManager",
    "CacheEntry", 
    "CacheStats",
    "CacheBackend",
    "MemoryCache",
    "DiskCache",
]
