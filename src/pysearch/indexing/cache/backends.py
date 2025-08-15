"""
Cache backend implementations for pysearch.

This module provides different storage backends for the caching system,
including in-memory and persistent disk-based storage options.

Classes:
    CacheBackend: Abstract base class for cache backends
    MemoryCache: In-memory cache with LRU eviction
    DiskCache: Persistent disk-based cache with compression

Features:
    - Multiple storage backends (memory, disk)
    - LRU eviction policies
    - Optional compression for disk storage
    - Thread-safe operations
    - Automatic cleanup and maintenance
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .models import CacheEntry
from ...utils.logging_config import get_logger


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
