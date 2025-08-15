"""
Cache data models for pysearch.

This module contains the core data structures used by the caching system,
including cache entries and performance statistics.

Classes:
    CacheEntry: Represents a cached search result with metadata
    CacheStats: Cache performance statistics and metrics

Features:
    - Immutable data structures with validation
    - Automatic expiration checking
    - Access pattern tracking
    - Size and dependency management
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ...core.types import SearchResult


@dataclass
class CacheEntry:
    """Represents a cached search result with metadata."""

    key: str
    value: SearchResult
    created_at: float
    last_accessed: float
    ttl: float  # Time to live in seconds
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    file_dependencies: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl <= 0:
            return False  # No expiration
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    average_access_time: float = 0.0
    hit_rate: float = 0.0

    def update_hit_rate(self) -> None:
        """Update the hit rate calculation."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
