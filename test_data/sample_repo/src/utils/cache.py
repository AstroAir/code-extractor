"""Caching utilities with decorator-based API.

Provides an in-memory cache with TTL expiration and
a decorator for transparent function result caching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry with expiration tracking."""

    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 minutes default

    @property
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl


class InMemoryCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key to look up.

        Returns:
            Cached value or None if not found/expired.
        """
        entry = self._store.get(key)
        if entry is None:
            self._miss_count += 1
            return None
        if entry.is_expired:
            del self._store[key]
            self._miss_count += 1
            return None
        self._hit_count += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in cache.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Time-to-live in seconds, or None for default.
        """
        self._store[key] = CacheEntry(
            value=value,
            ttl=ttl if ttl is not None else self.default_ttl,
        )

    def delete(self, key: str) -> bool:
        """Remove a key from cache.

        Returns:
            True if the key was found and removed.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries removed.
        """
        count = len(self._store)
        self._store.clear()
        return count

    @property
    def size(self) -> int:
        """Get the number of entries in cache."""
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total


# Global cache instance
_global_cache = InMemoryCache()


def cached(ttl: float = 300.0) -> Callable[..., Any]:
    """Decorator for caching function results.

    Args:
        ttl: Cache time-to-live in seconds.

    Returns:
        Decorator function.

    Example:
        @cached(ttl=60)
        async def get_user(user_id: str) -> User:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            cache_key = f"{func.__qualname__}:{args}:{kwargs}"
            result = _global_cache.get(cache_key)
            if result is not None:
                return result
            result = await func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            cache_key = f"{func.__qualname__}:{args}:{kwargs}"
            result = _global_cache.get(cache_key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl)
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def clear_cache() -> int:
    """Clear the global cache.

    Returns:
        Number of entries cleared.
    """
    return _global_cache.clear()
