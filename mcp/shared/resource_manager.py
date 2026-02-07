"""
Enhanced resource management for MCP server with caching and analytics.

Provides intelligent resource management with LRU caching, analytics tracking,
and memory optimization for the MCP server.
"""

import json
import sys
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any


class CacheEntry:
    """Represents a cache entry with metadata."""

    def __init__(self, value: Any, ttl: float | None = None) -> None:
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.expires_at = self.created_at + ttl if ttl else None

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def access(self) -> Any:
        """Access the cache entry and update metadata."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value


class ResourceManager:
    """Enhanced resource manager with caching and analytics."""

    def __init__(self, max_cache_size: int = 100, default_ttl: float = 300.0) -> None:
        """Initialize the resource manager.

        Args:
            max_cache_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl

        # Use OrderedDict for LRU functionality
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Analytics tracking
        self._analytics: dict[str, int | float] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_sets": 0,
            "cache_evictions": 0,
            "total_requests": 0,
            "start_time": time.time(),
        }

        # Access history for advanced analytics
        self._access_history: list[dict[str, Any]] = []

    def get_cache(self, key: str) -> Any | None:
        """Get a value from the cache."""
        self._analytics["total_requests"] += 1

        if key not in self._cache:
            self._analytics["cache_misses"] += 1
            return None

        entry = self._cache[key]

        # Check if expired
        if entry.is_expired():
            self._analytics["cache_misses"] += 1
            del self._cache[key]
            return None

        # Move to end (most recently used) and access
        self._cache.move_to_end(key)
        value = entry.access()
        self._analytics["cache_hits"] += 1

        # Track access
        self._track_access(key, "hit")

        return value

    def set_cache(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set a value in the cache with optional TTL."""
        if ttl is None:
            ttl = self.default_ttl

        # Remove existing entry if present
        if key in self._cache:
            del self._cache[key]

        # Check if we need to evict entries
        while len(self._cache) >= self.max_cache_size:
            # Remove least recently used (first item)
            oldest_key, _ = self._cache.popitem(last=False)
            self._analytics["cache_evictions"] += 1
            self._track_access(oldest_key, "eviction")

        # Add new entry
        self._cache[key] = CacheEntry(value, ttl)
        self._analytics["cache_sets"] += 1
        self._track_access(key, "set")

    def has_cache(self, key: str) -> bool:
        """Check if a key exists in cache and is not expired."""
        if key not in self._cache:
            return False

        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            return False

        return True

    def delete_cache(self, key: str) -> bool:
        """Delete a key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._track_access(key, "delete")
            return True
        return False

    def clear_cache(self) -> int:
        """Clear all cache entries and return count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        self._track_access("*", f"clear_{count}")
        return count

    def clean_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        expired_keys: list[str] = []
        current_time: float = time.time()

        for key, entry in self._cache.items():
            if entry.expires_at and current_time > entry.expires_at:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            self._track_access(key, "expired")

        return len(expired_keys)

    def get_cache_analytics(self) -> dict[str, Any]:
        """Get comprehensive cache analytics."""
        current_time: float = time.time()
        uptime: float = current_time - float(self._analytics["start_time"])

        total_requests: int = int(self._analytics["total_requests"])
        hit_rate: float = (
            (float(self._analytics["cache_hits"]) / total_requests) if total_requests > 0 else 0.0
        )

        # Cache size distribution
        cache_sizes: list[int] = []
        for entry in self._cache.values():
            try:
                # Estimate size of cached value
                size = len(json.dumps(entry.value, default=str))
                cache_sizes.append(size)
            except (TypeError, ValueError):
                cache_sizes.append(len(str(entry.value)))

        analytics: dict[str, Any] = {
            "total_requests": total_requests,
            "cache_hits": int(self._analytics["cache_hits"]),
            "cache_misses": int(self._analytics["cache_misses"]),
            "cache_sets": int(self._analytics["cache_sets"]),
            "cache_evictions": int(self._analytics["cache_evictions"]),
            "hit_rate": hit_rate,
            "current_size": len(self._cache),
            "max_size": self.max_cache_size,
            "uptime_seconds": uptime,
            "requests_per_second": total_requests / uptime if uptime > 0 else 0,
            "cache_size_stats": {
                "total_entries": len(cache_sizes),
                "total_size_bytes": sum(cache_sizes),
                "avg_entry_size_bytes": sum(cache_sizes) / len(cache_sizes) if cache_sizes else 0,
                "max_entry_size_bytes": max(cache_sizes) if cache_sizes else 0,
                "min_entry_size_bytes": min(cache_sizes) if cache_sizes else 0,
            },
        }

        # Recent access patterns
        recent_accesses: list[dict[str, Any]] = self._access_history[-100:]  # Last 100 accesses
        if recent_accesses:
            analytics["recent_activity"] = {
                "total_accesses": len(recent_accesses),
                "unique_keys": len(set(access["key"] for access in recent_accesses)),
                "access_types": {},
            }

            # Count access types
            for access in recent_accesses:
                access_type = access["type"]
                analytics["recent_activity"]["access_types"][access_type] = (
                    analytics["recent_activity"]["access_types"].get(access_type, 0) + 1
                )

        return analytics

    def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        cache_memory: int = 0

        for entry in self._cache.values():
            try:
                # Estimate memory usage
                size: int = sys.getsizeof(entry.value)
                size += sys.getsizeof(entry)
                cache_memory += size
            except (TypeError, AttributeError):
                cache_memory += 100  # Rough estimate

        return {
            "cache_memory_mb": cache_memory / (1024 * 1024),
            "total_entries": len(self._cache),
            "avg_entry_size_bytes": cache_memory / len(self._cache) if self._cache else 0,
            "analytics_memory_mb": sys.getsizeof(self._analytics) / (1024 * 1024),
            "history_memory_mb": sys.getsizeof(self._access_history) / (1024 * 1024),
        }

    def optimize_cache(self) -> dict[str, int]:
        """Optimize cache by removing expired entries and least used entries."""
        stats: dict[str, int] = {"expired_removed": 0, "lru_removed": 0}

        # Remove expired entries
        stats["expired_removed"] = self.clean_expired()

        # If cache is still too full, remove least recently used entries
        target_size: int = int(self.max_cache_size * 0.8)  # Keep at 80% capacity

        while len(self._cache) > target_size:
            # Remove least recently used (first item)
            oldest_key: str
            oldest_key, _ = self._cache.popitem(last=False)
            stats["lru_removed"] += 1
            self._analytics["cache_evictions"] += 1
            self._track_access(oldest_key, "optimization_eviction")

        return stats

    def get_cache_keys(self) -> list[str]:
        """Get list of all cache keys."""
        return list(self._cache.keys())

    def get_cache_info(self, key: str) -> dict[str, Any] | None:
        """Get detailed information about a cache entry."""
        if key not in self._cache:
            return None

        entry: CacheEntry = self._cache[key]

        return {
            "key": key,
            "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
            "last_accessed": datetime.fromtimestamp(entry.last_accessed).isoformat(),
            "access_count": entry.access_count,
            "expires_at": (
                datetime.fromtimestamp(entry.expires_at).isoformat() if entry.expires_at else None
            ),
            "is_expired": entry.is_expired(),
            "value_type": type(entry.value).__name__,
            "estimated_size_bytes": sys.getsizeof(entry.value),
        }

    def _track_access(self, key: str, access_type: str) -> None:
        """Track cache access for analytics."""
        # Limit history size to prevent memory bloat
        if len(self._access_history) > 1000:
            self._access_history = self._access_history[-500:]  # Keep last 500

        self._access_history.append({"key": key, "type": access_type, "timestamp": time.time()})

    def get_health_status(self) -> dict[str, Any]:
        """Get overall health status of the resource manager."""
        analytics: dict[str, Any] = self.get_cache_analytics()
        memory: dict[str, Any] = self.get_memory_usage()

        # Determine health status
        health_issues: list[str] = []

        # Check hit rate
        if analytics["hit_rate"] < 0.3:
            health_issues.append("Low cache hit rate")

        # Check memory usage
        if memory["cache_memory_mb"] > 100:  # Over 100MB
            health_issues.append("High memory usage")

        # Check cache utilization
        utilization: float = analytics["current_size"] / analytics["max_size"]
        if utilization > 0.95:
            health_issues.append("Cache nearly full")

        # Check for excessive evictions
        if analytics["cache_evictions"] > analytics["cache_sets"] * 0.5:
            health_issues.append("High eviction rate")

        health_status: str = "healthy" if not health_issues else "warning"
        if len(health_issues) > 2:
            health_status = "critical"

        return {
            "overall_health": {"status": health_status, "issues": health_issues},
            "cache_health": {
                "hit_rate": analytics["hit_rate"],
                "utilization": utilization,
                "eviction_rate": analytics["cache_evictions"] / max(analytics["cache_sets"], 1),
            },
            "memory_health": {
                "cache_memory_mb": memory["cache_memory_mb"],
                "total_entries": memory["total_entries"],
            },
            "performance_health": {
                "requests_per_second": analytics["requests_per_second"],
                "uptime_hours": analytics["uptime_seconds"] / 3600,
            },
            "diagnostics": {
                "last_optimization": "Not implemented",
                "next_cleanup": "On demand",
                "cache_efficiency": "Good" if analytics["hit_rate"] > 0.6 else "Poor",
            },
        }
