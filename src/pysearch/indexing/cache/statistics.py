"""
Cache statistics tracking and management.

This module provides comprehensive statistics tracking for cache operations,
including hit rates, access times, and performance metrics.

Classes:
    CacheStatistics: Manages cache performance statistics

Features:
    - Thread-safe statistics tracking
    - Hit/miss rate calculation
    - Access time monitoring
    - Performance metrics
    - Statistics reporting
"""

from __future__ import annotations

import threading
from typing import Any

from .models import CacheStats


class CacheStatistics:
    """
    Manages cache performance statistics and metrics.

    This class provides thread-safe tracking of cache operations and
    performance metrics for monitoring and optimization.
    """

    def __init__(self) -> None:
        self.stats = CacheStats()
        self._stats_lock = threading.RLock()

    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._stats_lock:
            self.stats.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._stats_lock:
            self.stats.misses += 1

    def record_eviction(self, count: int = 1) -> None:
        """
        Record cache evictions.

        Args:
            count: Number of entries evicted
        """
        with self._stats_lock:
            self.stats.evictions += count

    def record_invalidation(self, count: int = 1) -> None:
        """
        Record cache invalidations.

        Args:
            count: Number of entries invalidated
        """
        with self._stats_lock:
            self.stats.invalidations += count

    def update_access_time(self, elapsed: float) -> None:
        """
        Update average access time with a new measurement.

        Args:
            elapsed: Time taken for the cache operation
        """
        with self._stats_lock:
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.average_access_time = (
                    self.stats.average_access_time * (total_requests - 1) + elapsed
                ) / total_requests

    def update_entry_count(self, count: int) -> None:
        """
        Update the total number of cache entries.

        Args:
            count: Current number of cache entries
        """
        with self._stats_lock:
            self.stats.total_entries = count

    def update_size(self, size_bytes: int) -> None:
        """
        Update the total cache size.

        Args:
            size_bytes: Current cache size in bytes
        """
        with self._stats_lock:
            self.stats.total_size_bytes = size_bytes

    def add_size(self, size_bytes: int) -> None:
        """
        Add to the total cache size.

        Args:
            size_bytes: Size to add in bytes
        """
        with self._stats_lock:
            self.stats.total_size_bytes += size_bytes

    def subtract_size(self, size_bytes: int) -> None:
        """
        Subtract from the total cache size.

        Args:
            size_bytes: Size to subtract in bytes
        """
        with self._stats_lock:
            self.stats.total_size_bytes = max(0, self.stats.total_size_bytes - size_bytes)

    def get_hit_rate(self) -> float:
        """
        Calculate and return the current hit rate.

        Returns:
            Hit rate as a percentage (0.0 to 100.0)
        """
        with self._stats_lock:
            total_requests = self.stats.hits + self.stats.misses
            if total_requests == 0:
                return 0.0
            return (self.stats.hits / total_requests) * 100.0

    def get_stats_dict(self, additional_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Get all statistics as a dictionary.

        Args:
            additional_stats: Additional statistics to include

        Returns:
            Dictionary containing all cache statistics
        """
        with self._stats_lock:
            # Update hit rate before returning
            self.stats.update_hit_rate()

            stats_dict = {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate,
                "evictions": self.stats.evictions,
                "invalidations": self.stats.invalidations,
                "total_entries": self.stats.total_entries,
                "total_size_bytes": self.stats.total_size_bytes,
                "average_access_time": self.stats.average_access_time,
            }

            if additional_stats:
                stats_dict.update(additional_stats)

            return stats_dict

    def reset_stats(self) -> None:
        """Reset all statistics to zero."""
        with self._stats_lock:
            self.stats = CacheStats()

    def get_performance_summary(self) -> str:
        """
        Get a human-readable performance summary.

        Returns:
            Formatted string with key performance metrics
        """
        with self._stats_lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = self.get_hit_rate()

            return (
                f"Cache Performance: "
                f"{total_requests} requests, "
                f"{hit_rate:.1f}% hit rate, "
                f"{self.stats.total_entries} entries, "
                f"{self.stats.total_size_bytes / 1024 / 1024:.1f} MB"
            )
