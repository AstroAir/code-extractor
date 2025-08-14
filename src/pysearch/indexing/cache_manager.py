"""
Search result caching module for pysearch.

This module provides comprehensive caching capabilities including:
- Persistent search result caching with configurable TTL
- Intelligent cache invalidation based on file changes
- Memory and disk-based cache backends
- Cache compression and serialization
- Cache statistics and management
- LRU eviction policies

Classes:
    CacheEntry: Represents a cached search result
    CacheBackend: Abstract base for cache backends
    MemoryCache: In-memory cache implementation
    DiskCache: Persistent disk-based cache
    CacheManager: Main cache management interface

Features:
    - Multiple cache backends (memory, disk, hybrid)
    - Automatic cache invalidation on file changes
    - Configurable TTL and size limits
    - Cache compression for large results
    - Cache statistics and monitoring
    - Thread-safe operations

Example:
    Basic caching:
        >>> from pysearch.cache_manager import CacheManager
        >>> cache = CacheManager()
        >>>
        >>> # Cache search results
        >>> cache.set("query_key", search_result, ttl=3600)
        >>>
        >>> # Retrieve cached results
        >>> cached_result = cache.get("query_key")
        >>> if cached_result:
        ...     print("Using cached results")

    Advanced cache management:
        >>> # Configure cache with custom settings
        >>> cache = CacheManager(
        ...     backend="disk",
        ...     max_size=1000,
        ...     default_ttl=1800,
        ...     compression=True
        ... )
        >>>
        >>> # Monitor cache performance
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
"""

from __future__ import annotations

import builtins
import hashlib
import json
import pickle
import threading
import time
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging_config import get_logger
from ..core.types import SearchResult


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


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        pass

    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set a cache entry."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        """Get all cache keys."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get the number of cache entries."""
        pass


class MemoryCache(CacheBackend):
    """
    In-memory cache implementation with LRU eviction.

    Provides fast access to cached results but data is lost when
    the process terminates.
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size_bytes = 0

    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                return entry
            return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set a cache entry."""
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes
                del self._cache[key]

            # Check memory limits
            if (self._current_size_bytes + entry.size_bytes > self.max_memory_bytes or
                len(self._cache) >= self.max_size):
                self._evict_entries()

            # Add new entry
            self._cache[key] = entry
            self._current_size_bytes += entry.size_bytes
            return True

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    def size(self) -> int:
        """Get the number of cache entries."""
        return len(self._cache)

    def _evict_entries(self) -> None:
        """Evict least recently used entries to make space."""
        # Remove oldest entries until we're under limits
        while (len(self._cache) >= self.max_size or
               self._current_size_bytes > self.max_memory_bytes * 0.8):
            if not self._cache:
                break

            # Remove least recently used (first item)
            key, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= entry.size_bytes


class DiskCache(CacheBackend):
    """
    Persistent disk-based cache implementation.

    Stores cache entries on disk for persistence across process restarts.
    Uses pickle for serialization and optional compression.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        max_size: int = 10000,
        compression: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.compression = compression
        self._lock = threading.RLock()
        self.logger = get_logger()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file for metadata
        self.index_file = self.cache_dir / "cache_index.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        with self._lock:
            if key not in self._index:
                return None

            try:
                entry_file = self._get_entry_file(key)
                if not entry_file.exists():
                    # Clean up stale index entry
                    del self._index[key]
                    self._save_index()
                    return None

                # Load entry from disk
                with open(entry_file, 'rb') as f:
                    data = f.read()

                if self.compression:
                    data = zlib.decompress(data)

                entry: CacheEntry = pickle.loads(data)
                entry.touch()

                # Update index
                self._index[key]['last_accessed'] = entry.last_accessed
                self._index[key]['access_count'] = entry.access_count
                self._save_index()

                return entry

            except Exception as e:
                self.logger.error(f"Error loading cache entry {key}: {e}")
                return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set a cache entry."""
        with self._lock:
            try:
                # Check size limits and evict if necessary
                if len(self._index) >= self.max_size:
                    self._evict_entries()

                # Serialize entry
                data = pickle.dumps(entry)
                if self.compression:
                    data = zlib.compress(data)

                # Save to disk
                entry_file = self._get_entry_file(key)
                with open(entry_file, 'wb') as f:
                    f.write(data)

                # Update index
                self._index[key] = {
                    'created_at': entry.created_at,
                    'last_accessed': entry.last_accessed,
                    'ttl': entry.ttl,
                    'size_bytes': len(data),
                    'access_count': entry.access_count
                }
                self._save_index()

                return True

            except Exception as e:
                self.logger.error(f"Error saving cache entry {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        with self._lock:
            if key not in self._index:
                return False

            try:
                entry_file = self._get_entry_file(key)
                if entry_file.exists():
                    entry_file.unlink()

                del self._index[key]
                self._save_index()
                return True

            except Exception as e:
                self.logger.error(f"Error deleting cache entry {key}: {e}")
                return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            try:
                # Remove all entry files
                for entry_file in self.cache_dir.glob("entry_*.pkl"):
                    entry_file.unlink()

                self._index.clear()
                self._save_index()

            except Exception as e:
                self.logger.error(f"Error clearing cache: {e}")

    def keys(self) -> list[str]:
        """Get all cache keys."""
        return list(self._index.keys())

    def size(self) -> int:
        """Get the number of cache entries."""
        return len(self._index)

    def _get_entry_file(self, key: str) -> Path:
        """Get the file path for a cache entry."""
        # Use hash of key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"entry_{key_hash}.pkl"

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file) as f:
                    self._index = json.load(f)
        except Exception as e:
            self.logger.warning(f"Error loading cache index: {e}")
            self._index = {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache index: {e}")

    def _evict_entries(self) -> None:
        """Evict least recently used entries to make space."""
        # Sort by last accessed time
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1]['last_accessed']
        )

        # Remove oldest 20% of entries
        num_to_remove = max(1, len(sorted_entries) // 5)
        for key, _ in sorted_entries[:num_to_remove]:
            self.delete(key)


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
            dependent_keys = self._file_dependencies.get(file_path, set()).copy()

        invalidated = 0
        for key in dependent_keys:
            if self.delete(key):
                invalidated += 1

        if invalidated > 0:
            with self._stats_lock:
                self.stats.invalidations += invalidated

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
            with self._stats_lock:
                self.stats.invalidations += invalidated

            self.logger.debug(f"Invalidated {invalidated} cache entries matching pattern: {pattern}")

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

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
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
