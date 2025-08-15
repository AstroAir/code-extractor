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
import threading
import time
from pathlib import Path
from typing import Any

from .backends import CacheBackend, DiskCache, MemoryCache
from .models import CacheEntry, CacheStats
from ...core.types import SearchResult
from ...utils.logging_config import get_logger


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
        cleanup_interval: float = 300  # 5 minutes
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
                cache_dir=cache_dir,
                max_size=max_size,
                compression=compression
            )
        else:
            raise ValueError(f"Unknown cache backend: {backend}")

        # Statistics
        self.stats = CacheStats()
        self._stats_lock = threading.RLock()

        # File dependency tracking
        self._file_dependencies: dict[str, set[str]] = {}  # file -> cache_keys
        self._dependency_lock = threading.RLock()

        # Cleanup thread
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_stop_event = threading.Event()

        if auto_cleanup:
            self._start_cleanup_thread()

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
                self._record_miss()
                return None

            # Check expiration
            if entry.is_expired:
                self.backend.delete(key)
                self._remove_file_dependencies(key)
                self._record_miss()
                return None

            self._record_hit()
            return entry.value

        except Exception as e:
            self.logger.error(f"Error getting cache entry {key}: {e}")
            self._record_miss()
            return None

        finally:
            elapsed = time.time() - start_time
            self._update_access_time(elapsed)

    def set(
        self,
        key: str,
        value: SearchResult,
        ttl: float | None = None,
        file_dependencies: builtins.set[str] | None = None
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
                file_dependencies=file_dependencies or set()
            )

            # Store in backend
            success = self.backend.set(key, entry)

            if success:
                # Update file dependencies
                if file_dependencies:
                    self._add_file_dependencies(key, file_dependencies)

                # Update stats
                with self._stats_lock:
                    self.stats.total_entries = self.backend.size()
                    self.stats.total_size_bytes += size_bytes

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
            success = self.backend.delete(key)
            if success:
                self._remove_file_dependencies(key)
                with self._stats_lock:
                    self.stats.total_entries = self.backend.size()
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
        with self._dependency_lock:
            dependent_keys = self._file_dependencies.get(
                file_path, set()).copy()

        invalidated = 0
        for key in dependent_keys:
            if self.delete(key):
                invalidated += 1

        if invalidated > 0:
            with self._stats_lock:
                self.stats.invalidations += invalidated

            self.logger.debug(
                f"Invalidated {invalidated} cache entries for file: {file_path}")

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
            with self._stats_lock:
                self.stats.invalidations += invalidated

            self.logger.debug(
                f"Invalidated {invalidated} cache entries matching pattern: {pattern}")

        return invalidated

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            self.backend.clear()

            with self._dependency_lock:
                self._file_dependencies.clear()

            with self._stats_lock:
                self.stats.total_entries = 0
                self.stats.total_size_bytes = 0

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
        current_time = time.time()

        for key in self.backend.keys():
            try:
                entry = self.backend.get(key)
                if entry and entry.is_expired:
                    if self.backend.delete(key):
                        self._remove_file_dependencies(key)
                        removed += 1
            except Exception as e:
                self.logger.error(f"Error checking expiration for {key}: {e}")

        if removed > 0:
            with self._stats_lock:
                self.stats.evictions += removed
                self.stats.total_entries = self.backend.size()

            self.logger.debug(f"Cleaned up {removed} expired cache entries")

        return removed

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache performance statistics
        """
        with self._stats_lock:
            self.stats.update_hit_rate()
            self.stats.total_entries = self.backend.size()

            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate,
                "evictions": self.stats.evictions,
                "invalidations": self.stats.invalidations,
                "total_entries": self.stats.total_entries,
                "total_size_bytes": self.stats.total_size_bytes,
                "average_access_time": self.stats.average_access_time,
                "file_dependencies": len(self._file_dependencies)
            }

    def _calculate_size(self, value: SearchResult) -> int:
        """Calculate the approximate size of a search result."""
        try:
            # Use pickle to get a rough size estimate
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return len(str(value)) * 2  # Rough estimate

    def _add_file_dependencies(self, key: str, file_paths: builtins.set[str]) -> None:
        """Add file dependencies for a cache key."""
        with self._dependency_lock:
            for file_path in file_paths:
                if file_path not in self._file_dependencies:
                    self._file_dependencies[file_path] = set()
                self._file_dependencies[file_path].add(key)

    def _remove_file_dependencies(self, key: str) -> None:
        """Remove file dependencies for a cache key."""
        with self._dependency_lock:
            files_to_remove = []
            for file_path, keys in self._file_dependencies.items():
                keys.discard(key)
                if not keys:
                    files_to_remove.append(file_path)

            for file_path in files_to_remove:
                del self._file_dependencies[file_path]

    def _record_hit(self) -> None:
        """Record a cache hit."""
        with self._stats_lock:
            self.stats.hits += 1

    def _record_miss(self) -> None:
        """Record a cache miss."""
        with self._stats_lock:
            self.stats.misses += 1

    def _update_access_time(self, elapsed: float) -> None:
        """Update average access time."""
        with self._stats_lock:
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.average_access_time = (
                    (self.stats.average_access_time * (total_requests - 1) + elapsed) /
                    total_requests
                )

    def _start_cleanup_thread(self) -> None:
        """Start the automatic cleanup thread."""
        def cleanup_worker() -> None:
            while not self._cleanup_stop_event.wait(self.cleanup_interval):
                try:
                    self.cleanup_expired()
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {e}")

        self._cleanup_thread = threading.Thread(
            target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def shutdown(self) -> None:
        """Shutdown the cache manager and cleanup resources."""
        if self._cleanup_thread:
            self._cleanup_stop_event.set()
            self._cleanup_thread.join(timeout=5.0)

        self.logger.info("Cache manager shutdown complete")

    def __enter__(self) -> CacheManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()
