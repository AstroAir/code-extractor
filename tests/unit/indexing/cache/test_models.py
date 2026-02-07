"""Tests for pysearch.indexing.cache.models module."""

from __future__ import annotations

import time

import pytest

from pysearch.core.types import SearchResult, SearchStats
from pysearch.indexing.cache.models import CacheEntry, CacheStats


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def _make_result(self) -> SearchResult:
        return SearchResult(items=[], stats=SearchStats())

    def test_creation(self):
        now = time.time()
        entry = CacheEntry(
            key="test_key",
            value=self._make_result(),
            created_at=now,
            last_accessed=now,
            ttl=3600,
        )
        assert entry.key == "test_key"
        assert entry.ttl == 3600
        assert entry.access_count == 0

    def test_is_expired_not_expired(self):
        now = time.time()
        entry = CacheEntry(
            key="k", value=self._make_result(),
            created_at=now, last_accessed=now, ttl=3600,
        )
        assert entry.is_expired is False

    def test_is_expired_expired(self):
        old_time = time.time() - 7200
        entry = CacheEntry(
            key="k", value=self._make_result(),
            created_at=old_time, last_accessed=old_time, ttl=3600,
        )
        assert entry.is_expired is True

    def test_is_expired_no_ttl(self):
        old_time = time.time() - 999999
        entry = CacheEntry(
            key="k", value=self._make_result(),
            created_at=old_time, last_accessed=old_time, ttl=0,
        )
        assert entry.is_expired is False

    def test_age_seconds(self):
        now = time.time()
        entry = CacheEntry(
            key="k", value=self._make_result(),
            created_at=now - 100, last_accessed=now, ttl=3600,
        )
        assert entry.age_seconds >= 99

    def test_defaults(self):
        entry = CacheEntry(
            key="k", value=self._make_result(),
            created_at=0.0, last_accessed=0.0, ttl=0.0,
        )
        assert entry.size_bytes == 0
        assert entry.compressed is False
        assert entry.file_dependencies == set()
        assert entry.metadata == {}
