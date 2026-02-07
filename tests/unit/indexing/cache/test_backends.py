"""Tests for pysearch.indexing.cache.backends module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.core.types import SearchResult, SearchStats
from pysearch.indexing.cache.backends import CacheBackend, DiskCache, MemoryCache
from pysearch.indexing.cache.models import CacheEntry


def _make_entry(key: str = "test", ttl: float = 3600) -> CacheEntry:
    now = time.time()
    return CacheEntry(
        key=key,
        value=SearchResult(items=[], stats=SearchStats()),
        created_at=now,
        last_accessed=now,
        ttl=ttl,
    )


class TestMemoryCache:
    """Tests for MemoryCache class."""

    def test_init(self):
        cache = MemoryCache(max_size=100)
        assert cache is not None

    def test_set_and_get(self):
        cache = MemoryCache(max_size=100)
        entry = _make_entry("k1")
        cache.set("k1", entry)
        result = cache.get("k1")
        assert result is not None
        assert result.key == "k1"

    def test_get_nonexistent(self):
        cache = MemoryCache(max_size=100)
        assert cache.get("nonexistent") is None

    def test_delete(self):
        cache = MemoryCache(max_size=100)
        cache.set("k1", _make_entry("k1"))
        assert cache.delete("k1") is True
        assert cache.get("k1") is None

    def test_delete_nonexistent(self):
        cache = MemoryCache(max_size=100)
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        cache = MemoryCache(max_size=100)
        cache.set("k1", _make_entry("k1"))
        cache.set("k2", _make_entry("k2"))
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_lru_eviction(self):
        cache = MemoryCache(max_size=2)
        cache.set("k1", _make_entry("k1"))
        cache.set("k2", _make_entry("k2"))
        cache.set("k3", _make_entry("k3"))
        # k1 should be evicted
        assert cache.get("k1") is None
        assert cache.get("k3") is not None


class TestDiskCache:
    """Tests for DiskCache class."""

    def test_init(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        assert cache is not None

    def test_set_and_get(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        entry = _make_entry("dk1")
        cache.set("dk1", entry)
        result = cache.get("dk1")
        assert result is not None
        assert result.key == "dk1"

    def test_get_nonexistent(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        assert cache.get("nonexistent") is None

    def test_delete(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        cache.set("dk1", _make_entry("dk1"))
        assert cache.delete("dk1") is True
        assert cache.get("dk1") is None
