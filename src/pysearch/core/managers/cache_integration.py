"""
Advanced caching management for search operations.

This module provides comprehensive caching capabilities for search results,
file contents, and other expensive operations to improve performance.

Classes:
    CacheIntegrationManager: Manages advanced caching functionality

Key Features:
    - Search result caching with file dependency tracking
    - Configurable cache backends (memory, disk)
    - Automatic cache invalidation on file changes
    - Cache statistics and management
    - TTL-based expiration

Example:
    Using cache integration:
        >>> from pysearch.core.managers.cache_integration import CacheIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = CacheIntegrationManager(config)
        >>> manager.enable_caching(backend="memory", max_size=1000)
        >>> # Caching is now active for search operations
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ...utils.helpers import read_text_safely
from ..config import SearchConfig
from ..types import Query, SearchResult


class CacheIntegrationManager:
    """Manages advanced caching functionality for search operations."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.cache_manager: Any = None
        self._caching_enabled = False

        # In-memory caches for backward compatibility
        # file content cache: path -> (mtime, access_time, content)
        self._file_content_cache: dict[Path, tuple[float, float, str]] = {}
        self._search_result_cache: dict[str, tuple[float, SearchResult]] = {}
        self._cache_lock = threading.RLock()
        self.cache_ttl: float = 300.0  # 5 minutes TTL

    def enable_caching(
        self,
        backend: str = "memory",
        cache_dir: Path | str | None = None,
        max_size: int = 1000,
        default_ttl: float = 3600,
        compression: bool = False,
    ) -> bool:
        """
        Enable search result caching for improved performance.

        Caching stores search results to avoid re-executing expensive searches.
        Results are automatically invalidated when files change (if file watching
        is enabled) or when they expire.

        Args:
            backend: Cache backend ("memory" or "disk")
            cache_dir: Directory for disk cache (required for disk backend)
            max_size: Maximum number of cached results
            default_ttl: Default cache time-to-live in seconds
            compression: Enable compression for cached results

        Returns:
            True if caching was enabled successfully, False otherwise
        """
        if self._caching_enabled:
            return True

        try:
            from ...indexing.cache import CacheManager

            if backend == "disk" and cache_dir is None:
                cache_dir = self.config.resolve_cache_dir() / "search_cache"

            self.cache_manager = CacheManager(
                backend=backend,
                cache_dir=Path(cache_dir) if cache_dir else None,
                max_size=max_size,
                default_ttl=default_ttl,
                compression=compression,
            )

            self._caching_enabled = True
            return True

        except Exception:
            return False

    def disable_caching(self) -> None:
        """Disable search result caching."""
        if not self._caching_enabled:
            return

        try:
            if self.cache_manager:
                self.cache_manager.clear()
                self.cache_manager = None
            self._caching_enabled = False
        except Exception:
            pass

    def is_caching_enabled(self) -> bool:
        """Check if caching is currently enabled."""
        return self._caching_enabled

    def get_cached_result(self, query: Query) -> SearchResult | None:
        """Get cached search result for a query."""
        if not self._caching_enabled or not self.cache_manager:
            return self._get_legacy_cached_result(query)

        try:
            cache_key = self._generate_cache_key(query)
            return self.cache_manager.get(cache_key)
        except Exception:
            return None

    def cache_result(self, query: Query, result: SearchResult) -> None:
        """Cache a search result."""
        if not self._caching_enabled:
            self._cache_legacy_result(query, result)
            return

        if not self.cache_manager:
            return

        try:
            cache_key = self._generate_cache_key(query)
            file_dependencies = {str(p) for p in self._get_file_dependencies(result)}
            self.cache_manager.set(key=cache_key, value=result, file_dependencies=file_dependencies)
        except Exception:
            pass

    def _generate_cache_key(self, query: Query) -> str:
        """Generate cache key for search query."""
        return (
            f"{query.pattern}:{query.use_regex}:{query.use_ast}:{query.context}:"
            f"{hash(str(query.filters))}:{hash(str(query.metadata_filters))}"
        )

    def _get_file_dependencies(self, result: SearchResult) -> list[Path]:
        """Extract file dependencies from search result."""
        return [item.file for item in result.items]

    def _get_legacy_cached_result(self, query: Query) -> SearchResult | None:
        """Get cached result using legacy cache system."""
        cache_key = self._generate_cache_key(query)

        with self._cache_lock:
            if cache_key in self._search_result_cache:
                timestamp, result = self._search_result_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    return result
                else:
                    # Remove expired entry
                    del self._search_result_cache[cache_key]

        return None

    def _cache_legacy_result(self, query: Query, result: SearchResult) -> None:
        """Cache result using legacy cache system."""
        cache_key = self._generate_cache_key(query)

        with self._cache_lock:
            self._search_result_cache[cache_key] = (time.time(), result)

            # Limit cache size
            if len(self._search_result_cache) > 100:
                oldest_keys = sorted(
                    self._search_result_cache.keys(), key=lambda k: self._search_result_cache[k][0]
                )[:20]
                for k in oldest_keys:
                    del self._search_result_cache[k]

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl

    def get_cached_file_content(self, path: Path) -> str | None:
        """
        Get file content with caching based on modification time.

        This method implements an in-memory cache for file contents to avoid
        repeatedly reading the same files. Cache entries are invalidated when
        the file's modification time changes. Uses LRU eviction based on
        access time when the cache exceeds its size limit.

        Args:
            path: Path to the file to read

        Returns:
            File content as string if successful, None if file cannot be read
            or exceeds size limits
        """
        try:
            stat = path.stat()
            current_mtime = stat.st_mtime

            with self._cache_lock:
                if path in self._file_content_cache:
                    cached_mtime, _access_time, content = self._file_content_cache[path]
                    if cached_mtime == current_mtime:
                        # Update access time for LRU tracking
                        self._file_content_cache[path] = (cached_mtime, time.time(), content)
                        return content

                # Cache miss or outdated - read file
                try:
                    file_content = read_text_safely(path, max_bytes=self.config.max_file_bytes)
                    if file_content is not None:
                        self._file_content_cache[path] = (current_mtime, time.time(), file_content)

                        # LRU eviction: remove least recently accessed entries
                        if len(self._file_content_cache) > 1000:
                            sorted_entries = sorted(
                                self._file_content_cache.items(),
                                key=lambda item: item[1][1],  # sort by access_time (oldest first)
                            )
                            for k, _ in sorted_entries[:200]:
                                del self._file_content_cache[k]

                        return file_content
                    else:
                        return None
                except Exception:
                    return None

        except Exception:
            return None

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        with self._cache_lock:
            self._file_content_cache.clear()
            self._search_result_cache.clear()

        if self.cache_manager:
            try:
                self.cache_manager.clear()
            except Exception:
                pass

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "enabled": self._caching_enabled,
            "backend": (
                "memory"
                if not self.cache_manager
                else getattr(self.cache_manager, "backend", "unknown")
            ),
            "file_content_cache_size": len(self._file_content_cache),
            "search_result_cache_size": len(self._search_result_cache),
        }

        if self.cache_manager:
            try:
                cache_stats = self.cache_manager.get_stats()
                stats.update(cache_stats)
            except Exception:
                pass

        return stats

    def invalidate_file_cache(self, file_path: Path) -> None:
        """Invalidate cache entries for a specific file."""
        with self._cache_lock:
            # Remove from file content cache
            if file_path in self._file_content_cache:
                del self._file_content_cache[file_path]

        # Invalidate search results that depend on this file
        if self.cache_manager:
            try:
                self.cache_manager.invalidate_by_file(str(file_path))
            except Exception:
                pass

    def set_cache_ttl(self, ttl: float) -> None:
        """Set cache time-to-live in seconds."""
        self.cache_ttl = ttl

        if self.cache_manager:
            try:
                self.cache_manager.set_default_ttl(ttl)
            except Exception:
                pass

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as a percentage."""
        if self.cache_manager:
            try:
                stats = self.cache_manager.get_stats()
                hits = stats.get("hits", 0)
                misses = stats.get("misses", 0)
                total = hits + misses
                return (hits / total * 100) if total > 0 else 0.0
            except Exception:
                pass

        return 0.0
