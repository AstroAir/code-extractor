"""
Cache manager for pysearch search results.

This module provides the main cache management interface with automatic
invalidation, statistics tracking, and configurable backends.

Classes:
    CacheManager: Main cache management interface

Features:
    - High-level caching operations
    - Automatic cache invalidation based on file changes
    - Statistics tracking and monitoring
    - Multiple backend support (memory, disk)
    - File dependency tracking
    - Automatic cleanup of expired entries
"""

from __future__ import annotations

import builtins
import pickle
import time
from pathlib import Path
from typing import Any

from ...core.types import SearchResult
from ...utils.logging_config import get_logger
from .backends import CacheBackend, DiskCache, MemoryCache
from .cleanup import CacheCleanup
from .dependencies import DependencyTracker
from .models import CacheEntry
from .statistics import CacheStatistics


class CacheManager:
    """
    Main cache management interface for PySearch.

    Provides high-level caching operations with automatic invalidation,
    statistics tracking, and configurable backends.
    """

    def __init__(
        self,
        backend: str = "memory",
        cache_dir: Path | str | None = None,
        max_size: int = 1000,
        default_ttl: float = 3600,  # 1 hour
        compression: bool = False,
        auto_cleanup: bool = True,
        cleanup_interval: float = 300,  # 5 minutes
    ):
        """
        Initialize cache manager.

        Args:
            backend: Cache backend type ("memory" or "disk")
            cache_dir: Directory for disk cache (required for disk backend)
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
            compression: Enable compression for cache entries
            auto_cleanup: Enable automatic cleanup of expired entries
            cleanup_interval: Interval between cleanup runs in seconds
        """
        self.default_ttl = default_ttl
        self.compression = compression
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        self.logger = get_logger()

        # Initialize backend
        if backend == "memory":
            self.backend: CacheBackend = MemoryCache(max_size=max_size)
        elif backend == "disk":
            if cache_dir is None:
                cache_dir = Path.home() / ".pysearch" / "cache"
            self.backend = DiskCache(
                cache_dir=cache_dir, max_size=max_size, compression=compression
            )
        else:
            raise ValueError(f"Unknown cache backend: {backend}")

        # Statistics tracking
        self.statistics = CacheStatistics()

        # File dependency tracking
        self.dependencies = DependencyTracker()

        # Cleanup management
        self.cleanup_manager = CacheCleanup(
            cleanup_callback=self.cleanup_expired,
            cleanup_interval=cleanup_interval,
            auto_cleanup=auto_cleanup,
        )

    def get(self, key: str) -> SearchResult | None:
        """
        Get a cached search result.

        Args:
            key: Cache key

        Returns:
            Cached SearchResult if found and valid, None otherwise
        """
        start_time = time.time()

        try:
            entry = self.backend.get(key)

            if entry is None:
                self.statistics.record_miss()
                return None

            # Check expiration
            if entry.is_expired:
                self.backend.delete(key)
                self.dependencies.remove_dependencies(key)
                self.statistics.record_miss()
                return None

            self.statistics.record_hit()
            return entry.value

        except Exception as e:
            self.logger.error(f"Error getting cache entry {key}: {e}")
            self.statistics.record_miss()
            return None

        finally:
            elapsed = time.time() - start_time
            self.statistics.update_access_time(elapsed)

    def set(
        self,
        key: str,
        value: SearchResult,
        ttl: float | None = None,
        file_dependencies: builtins.set[str] | None = None,
    ) -> bool:
        """
        Set a cached search result.

        Args:
            key: Cache key
            value: SearchResult to cache
            ttl: Time-to-live in seconds (uses default if None)
            file_dependencies: Set of file paths this result depends on

        Returns:
            True if cached successfully, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl

        try:
            # Calculate size
            size_bytes = self._calculate_size(value)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                size_bytes=size_bytes,
                compressed=self.compression,
                file_dependencies=file_dependencies or set(),
            )

            # Store in backend
            success = self.backend.set(key, entry)

            if success:
                # Update file dependencies
                if file_dependencies:
                    self.dependencies.add_dependencies(key, file_dependencies)

                # Update stats
                self.statistics.update_entry_count(self.backend.size())
                self.statistics.add_size(size_bytes)

            return success

        except Exception as e:
            self.logger.error(f"Error setting cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a cached entry.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Get entry size before deleting for accurate size tracking
            entry = self.backend.get(key)
            entry_size = entry.size_bytes if entry else 0

            success = self.backend.delete(key)
            if success:
                self.dependencies.remove_dependencies(key)
                self.statistics.update_entry_count(self.backend.size())
                if entry_size > 0:
                    self.statistics.subtract_size(entry_size)
            return success

        except Exception as e:
            self.logger.error(f"Error deleting cache entry {key}: {e}")
            return False

    def invalidate_by_file(self, file_path: str) -> int:
        """
        Invalidate all cache entries that depend on a specific file.

        Args:
            file_path: Path of the file that changed

        Returns:
            Number of cache entries invalidated
        """
        dependent_keys = self.dependencies.get_dependent_keys(file_path)

        invalidated = 0
        for key in dependent_keys:
            if self.delete(key):
                invalidated += 1

        if invalidated > 0:
            self.statistics.record_invalidation(invalidated)

            self.logger.debug(f"Invalidated {invalidated} cache entries for file: {file_path}")

        return invalidated

    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a key pattern.

        Args:
            pattern: Pattern to match against cache keys

        Returns:
            Number of cache entries invalidated
        """
        import fnmatch

        keys_to_delete = []
        for key in self.backend.keys():
            if fnmatch.fnmatch(key, pattern):
                keys_to_delete.append(key)

        invalidated = 0
        for key in keys_to_delete:
            if self.delete(key):
                invalidated += 1

        if invalidated > 0:
            self.statistics.record_invalidation(invalidated)

            self.logger.debug(
                f"Invalidated {invalidated} cache entries matching pattern: {pattern}"
            )

        return invalidated

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            self.backend.clear()
            self.dependencies.clear_all_dependencies()
            self.statistics.reset_stats()
            self.logger.info("Cache cleared")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        removed = 0

        for key in self.backend.keys():
            try:
                entry = self.backend.get(key)
                if entry and entry.is_expired:
                    if self.backend.delete(key):
                        self.dependencies.remove_dependencies(key)
                        removed += 1
            except Exception as e:
                self.logger.error(f"Error checking expiration for {key}: {e}")

        if removed > 0:
            self.statistics.record_eviction(removed)
            self.statistics.update_entry_count(self.backend.size())

            self.logger.debug(f"Cleaned up {removed} expired cache entries")

        return removed

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache performance statistics
        """
        self.statistics.update_entry_count(self.backend.size())

        # Recalculate total size from backend entries
        total_size = 0
        for k in self.backend.keys():
            entry = self.backend.get(k)
            if entry:
                total_size += entry.size_bytes
        self.statistics.update_size(total_size)

        # Clean up orphaned dependency entries
        self.dependencies.cleanup_empty_dependencies()

        additional_stats = {
            "file_dependencies": self.dependencies.get_dependency_count(),
            "tracked_files": len(self.dependencies.get_files_with_dependencies()),
            "performance_summary": self.statistics.get_performance_summary(),
        }

        return self.statistics.get_stats_dict(additional_stats)

    def _calculate_size(self, value: SearchResult) -> int:
        """Calculate the approximate size of a search result."""
        try:
            # Use pickle to get a rough size estimate
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return len(str(value)) * 2  # Rough estimate

    def set_default_ttl(self, ttl: float) -> None:
        """Update the default time-to-live for new cache entries.

        Args:
            ttl: New default TTL in seconds.
        """
        self.default_ttl = ttl

    def shutdown(self) -> None:
        """Shutdown the cache manager and cleanup resources."""
        self.cleanup_manager.shutdown()
        self.logger.info("Cache manager shutdown complete")

    def __enter__(self) -> CacheManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()
