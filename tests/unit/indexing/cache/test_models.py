"""Tests for pysearch.indexing.cache.models module."""

from __future__ import annotations

import time

import pytest

from pysearch.core.types import SearchResult, SearchStats
from pysearch.indexing.cache.models import CacheEntry, CacheStats


def _make_result() -> SearchResult:
    return SearchResult(items=[], stats=SearchStats())


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creation(self):
        now = time.time()
        entry = CacheEntry(
            key="test_key",
            value=_make_result(),
            created_at=now,
            last_accessed=now,
            ttl=3600,
        )
        assert entry.key == "test_key"
        assert entry.ttl == 3600
        assert entry.access_count == 0
        assert entry.size_bytes == 0
        assert entry.compressed is False
        assert entry.file_dependencies == set()
        assert entry.metadata == {}

    def test_is_expired_not_expired(self):
        now = time.time()
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=now,
            last_accessed=now,
            ttl=3600,
        )
        assert entry.is_expired is False

    def test_is_expired_expired(self):
        old_time = time.time() - 7200
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=old_time,
            last_accessed=old_time,
            ttl=3600,
        )
        assert entry.is_expired is True

    def test_is_expired_zero_ttl(self):
        old_time = time.time() - 999999
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=old_time,
            last_accessed=old_time,
            ttl=0,
        )
        assert entry.is_expired is False

    def test_is_expired_negative_ttl(self):
        old_time = time.time() - 999999
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=old_time,
            last_accessed=old_time,
            ttl=-1,
        )
        assert entry.is_expired is False

    def test_age_seconds(self):
        now = time.time()
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=now - 100,
            last_accessed=now,
            ttl=3600,
        )
        assert entry.age_seconds >= 99

    def test_touch(self):
        now = time.time()
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=now - 10,
            last_accessed=now - 10,
            ttl=3600,
        )
        assert entry.access_count == 0
        old_accessed = entry.last_accessed

        entry.touch()

        assert entry.access_count == 1
        assert entry.last_accessed >= old_accessed

        entry.touch()
        assert entry.access_count == 2

    def test_file_dependencies_mutable(self):
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=0.0,
            last_accessed=0.0,
            ttl=0.0,
            file_dependencies={"a.py", "b.py"},
        )
        assert entry.file_dependencies == {"a.py", "b.py"}

    def test_metadata(self):
        entry = CacheEntry(
            key="k",
            value=_make_result(),
            created_at=0.0,
            last_accessed=0.0,
            ttl=0.0,
            metadata={"source": "test"},
        )
        assert entry.metadata == {"source": "test"}


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_values(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.invalidations == 0
        assert stats.total_entries == 0
        assert stats.total_size_bytes == 0
        assert stats.average_access_time == 0.0
        assert stats.hit_rate == 0.0

    def test_update_hit_rate(self):
        stats = CacheStats(hits=3, misses=1)
        stats.update_hit_rate()
        assert stats.hit_rate == pytest.approx(0.75)

    def test_update_hit_rate_no_requests(self):
        stats = CacheStats()
        stats.update_hit_rate()
        assert stats.hit_rate == 0.0

    def test_update_hit_rate_all_hits(self):
        stats = CacheStats(hits=10, misses=0)
        stats.update_hit_rate()
        assert stats.hit_rate == pytest.approx(1.0)

    def test_update_hit_rate_all_misses(self):
        stats = CacheStats(hits=0, misses=10)
        stats.update_hit_rate()
        assert stats.hit_rate == pytest.approx(0.0)
