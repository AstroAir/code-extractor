"""
Tests for mcp/shared/resource_manager.py — ResourceManager and CacheEntry.

Covers: get_cache, set_cache, has_cache, delete_cache, clear_cache,
clean_expired, get_cache_analytics, get_memory_usage, optimize_cache,
get_cache_keys, get_cache_info, get_health_status, CacheEntry lifecycle,
and LRU eviction behaviour.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.shared.resource_manager import CacheEntry, ResourceManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resource_manager():
    """Create a fresh ResourceManager instance."""
    return ResourceManager()


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    """Tests for CacheEntry lifecycle."""

    def test_not_expired_without_ttl(self):
        """Entry without TTL never expires."""
        entry = CacheEntry("value", ttl=None)
        assert not entry.is_expired()

    def test_not_expired_within_ttl(self):
        """Entry within TTL is not expired."""
        entry = CacheEntry("value", ttl=300)
        assert not entry.is_expired()

    def test_access_updates_metadata(self):
        """access() increments access_count and updates last_accessed."""
        entry = CacheEntry("value")
        initial_access = entry.last_accessed
        val = entry.access()
        assert val == "value"
        assert entry.access_count == 1
        assert entry.last_accessed >= initial_access


# ---------------------------------------------------------------------------
# Basic cache operations
# ---------------------------------------------------------------------------


class TestCacheOperations:
    """Tests for basic get/set/has/delete/clear cache operations."""

    def test_set_and_get(self, resource_manager):
        """set_cache + get_cache round-trips correctly."""
        resource_manager.set_cache("key1", {"data": "value1"})
        cached = resource_manager.get_cache("key1")
        assert cached == {"data": "value1"}

    def test_get_nonexistent(self, resource_manager):
        """get_cache returns None for missing keys."""
        assert resource_manager.get_cache("missing") is None

    def test_has_cache_exists(self, resource_manager):
        """has_cache returns True for existing non-expired entries."""
        resource_manager.set_cache("key1", "val")
        assert resource_manager.has_cache("key1")

    def test_has_cache_missing(self, resource_manager):
        """has_cache returns False for missing keys."""
        assert not resource_manager.has_cache("missing")

    def test_delete_cache(self, resource_manager):
        """delete_cache removes the entry and returns True."""
        resource_manager.set_cache("key1", "val")
        assert resource_manager.delete_cache("key1")
        assert resource_manager.get_cache("key1") is None

    def test_delete_cache_missing(self, resource_manager):
        """delete_cache returns False for non-existent keys."""
        assert not resource_manager.delete_cache("missing")

    def test_clear_cache(self, resource_manager):
        """clear_cache removes all entries and returns count."""
        resource_manager.set_cache("k1", "v1")
        resource_manager.set_cache("k2", "v2")
        count = resource_manager.clear_cache()
        assert count == 2
        assert resource_manager.get_cache("k1") is None


# ---------------------------------------------------------------------------
# Cache expiration
# ---------------------------------------------------------------------------


class TestCacheExpiration:
    """Tests for cache TTL expiration."""

    @pytest.mark.asyncio
    async def test_expiration(self, resource_manager):
        """Entries with short TTL expire after the TTL period."""
        resource_manager.set_cache("exp_key", {"data": "expires"}, ttl=0.1)
        assert resource_manager.get_cache("exp_key") == {"data": "expires"}
        await asyncio.sleep(0.2)
        assert resource_manager.get_cache("exp_key") is None

    @pytest.mark.asyncio
    async def test_has_cache_expired(self, resource_manager):
        """has_cache returns False and cleans up expired entries."""
        resource_manager.set_cache("exp_key", "val", ttl=0.1)
        await asyncio.sleep(0.2)
        assert not resource_manager.has_cache("exp_key")


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    """Tests for LRU eviction when cache is full."""

    def test_eviction(self, resource_manager):
        """Oldest entries are evicted when cache reaches max_cache_size."""
        for i in range(100):
            resource_manager.set_cache(f"key_{i}", f"value_{i}")

        resource_manager.set_cache("key_new", "value_new")

        assert resource_manager.get_cache("key_0") is None
        assert resource_manager.get_cache("key_new") is not None

    def test_eviction_counter(self, resource_manager):
        """Eviction increments the eviction counter in analytics."""
        for i in range(101):
            resource_manager.set_cache(f"key_{i}", f"v_{i}")
        analytics = resource_manager.get_cache_analytics()
        assert analytics["cache_evictions"] >= 1


# ---------------------------------------------------------------------------
# clean_expired
# ---------------------------------------------------------------------------


class TestCleanExpired:
    """Tests for clean_expired."""

    @pytest.mark.asyncio
    async def test_clean_expired_removes_old(self, resource_manager):
        """clean_expired removes entries past their TTL."""
        resource_manager.set_cache("e1", "v1", ttl=0.1)
        resource_manager.set_cache("e2", "v2", ttl=0.1)
        resource_manager.set_cache("keep", "v3", ttl=300)
        await asyncio.sleep(0.2)
        cleaned = resource_manager.clean_expired()
        assert cleaned == 2
        assert resource_manager.get_cache("keep") is not None

    def test_clean_expired_none(self, resource_manager):
        """clean_expired returns 0 when nothing is expired."""
        resource_manager.set_cache("k1", "v1")
        cleaned = resource_manager.clean_expired()
        assert cleaned == 0


# ---------------------------------------------------------------------------
# Cache analytics
# ---------------------------------------------------------------------------


class TestCacheAnalytics:
    """Tests for get_cache_analytics."""

    def test_analytics_structure(self, resource_manager):
        """get_cache_analytics returns expected keys."""
        resource_manager.set_cache("k1", "v1")
        resource_manager.get_cache("k1")  # hit
        resource_manager.get_cache("miss")  # miss

        analytics = resource_manager.get_cache_analytics()
        assert analytics["total_requests"] >= 2
        assert analytics["cache_hits"] >= 1
        assert analytics["cache_misses"] >= 1
        assert 0 <= analytics["hit_rate"] <= 1
        assert "current_size" in analytics
        assert "max_size" in analytics


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------


class TestMemoryUsage:
    """Tests for get_memory_usage."""

    def test_memory_usage_structure(self, resource_manager):
        """get_memory_usage returns expected keys."""
        memory_info = resource_manager.get_memory_usage()
        assert isinstance(memory_info, dict)
        assert "cache_memory_mb" in memory_info
        assert "total_entries" in memory_info
        assert memory_info["cache_memory_mb"] >= 0


# ---------------------------------------------------------------------------
# optimize_cache
# ---------------------------------------------------------------------------


class TestOptimizeCache:
    """Tests for optimize_cache."""

    def test_optimize_reduces_size(self, resource_manager):
        """optimize_cache reduces cache to 80% of max when full."""
        for i in range(100):
            resource_manager.set_cache(f"key_{i}", f"v_{i}")
        stats = resource_manager.optimize_cache()
        assert isinstance(stats, dict)
        assert stats["lru_removed"] >= 0
        assert len(resource_manager._cache) <= int(100 * 0.8)


# ---------------------------------------------------------------------------
# get_cache_keys / get_cache_info
# ---------------------------------------------------------------------------


class TestCacheInfo:
    """Tests for get_cache_keys and get_cache_info."""

    def test_get_cache_keys(self, resource_manager):
        """get_cache_keys returns all current keys."""
        resource_manager.set_cache("a", 1)
        resource_manager.set_cache("b", 2)
        keys = resource_manager.get_cache_keys()
        assert set(keys) == {"a", "b"}

    def test_get_cache_info_exists(self, resource_manager):
        """get_cache_info returns metadata for an existing key."""
        resource_manager.set_cache("k1", "val")
        info = resource_manager.get_cache_info("k1")
        assert info is not None
        assert info["key"] == "k1"
        assert "created_at" in info
        assert "access_count" in info
        assert info["value_type"] == "str"

    def test_get_cache_info_missing(self, resource_manager):
        """get_cache_info returns None for missing keys."""
        assert resource_manager.get_cache_info("missing") is None


# ---------------------------------------------------------------------------
# get_health_status
# ---------------------------------------------------------------------------


class TestHealthStatus:
    """Tests for get_health_status."""

    def test_health_status_structure(self, resource_manager):
        """get_health_status returns expected keys."""
        health = resource_manager.get_health_status()
        assert isinstance(health, dict)
        assert "overall_health" in health
        assert "cache_health" in health
        assert "memory_health" in health
        assert "performance_health" in health
        assert health["overall_health"]["status"] in ("healthy", "warning", "critical")
