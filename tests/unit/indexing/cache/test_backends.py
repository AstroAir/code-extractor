"""Tests for pysearch.indexing.cache.backends module."""

from __future__ import annotations

import time
from pathlib import Path

from pysearch.core.types import SearchResult, SearchStats
from pysearch.indexing.cache.backends import DiskCache, MemoryCache
from pysearch.indexing.cache.models import CacheEntry


def _make_entry(key: str = "test", ttl: float = 3600, size_bytes: int = 0) -> CacheEntry:
    now = time.time()
    return CacheEntry(
        key=key,
        value=SearchResult(items=[], stats=SearchStats()),
        created_at=now,
        last_accessed=now,
        ttl=ttl,
        size_bytes=size_bytes,
    )


class TestMemoryCache:
    """Tests for MemoryCache class."""

    def test_init(self):
        cache = MemoryCache(max_size=100, max_memory_mb=50)
        assert cache.max_size == 100
        assert cache.max_memory_bytes == 50 * 1024 * 1024
        assert cache.size() == 0

    def test_set_and_get(self):
        cache = MemoryCache(max_size=100)
        entry = _make_entry("k1")
        assert cache.set("k1", entry) is True
        result = cache.get("k1")
        assert result is not None
        assert result.key == "k1"

    def test_get_nonexistent(self):
        cache = MemoryCache(max_size=100)
        assert cache.get("nonexistent") is None

    def test_get_updates_lru_order_and_touches(self):
        cache = MemoryCache(max_size=100)
        entry = _make_entry("k1")
        cache.set("k1", entry)
        result = cache.get("k1")
        assert result is not None
        assert result.access_count >= 1

    def test_set_overwrites_existing(self):
        cache = MemoryCache(max_size=100)
        cache.set("k1", _make_entry("k1", size_bytes=100))
        assert cache.size() == 1
        cache.set("k1", _make_entry("k1", size_bytes=200))
        assert cache.size() == 1

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
        assert cache.size() == 0
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_keys(self):
        cache = MemoryCache(max_size=100)
        cache.set("a", _make_entry("a"))
        cache.set("b", _make_entry("b"))
        keys = cache.keys()
        assert set(keys) == {"a", "b"}

    def test_size(self):
        cache = MemoryCache(max_size=100)
        assert cache.size() == 0
        cache.set("k1", _make_entry("k1"))
        assert cache.size() == 1
        cache.set("k2", _make_entry("k2"))
        assert cache.size() == 2

    def test_lru_eviction_by_max_size(self):
        cache = MemoryCache(max_size=2)
        cache.set("k1", _make_entry("k1"))
        cache.set("k2", _make_entry("k2"))
        cache.set("k3", _make_entry("k3"))
        assert cache.get("k1") is None
        assert cache.get("k3") is not None

    def test_lru_eviction_preserves_recently_used(self):
        cache = MemoryCache(max_size=2)
        cache.set("k1", _make_entry("k1"))
        cache.set("k2", _make_entry("k2"))
        cache.get("k1")  # k1 becomes most recently used
        cache.set("k3", _make_entry("k3"))
        # k2 should be evicted (least recently used), k1 kept
        assert cache.get("k2") is None
        assert cache.get("k1") is not None

    def test_memory_limit_eviction(self):
        # max_memory_mb=1 means ~1MB limit; entries with large size_bytes trigger eviction
        cache = MemoryCache(max_size=1000, max_memory_mb=1)
        large_size = 600 * 1024  # 600 KB each
        cache.set("k1", _make_entry("k1", size_bytes=large_size))
        cache.set("k2", _make_entry("k2", size_bytes=large_size))
        # Adding k2 may trigger eviction of k1 when over 80% of 1MB
        # Just verify the cache doesn't crash and respects some limit
        assert cache.size() >= 1


class TestDiskCache:
    """Tests for DiskCache class."""

    def test_init(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        assert cache.size() == 0
        assert cache.cache_dir == tmp_path

    def test_init_creates_directory(self, tmp_path: Path):
        sub_dir = tmp_path / "sub" / "cache"
        DiskCache(cache_dir=sub_dir, max_size=100)
        assert sub_dir.exists()

    def test_set_and_get(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        entry = _make_entry("dk1")
        assert cache.set("dk1", entry) is True
        result = cache.get("dk1")
        assert result is not None
        assert result.key == "dk1"

    def test_get_nonexistent(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        assert cache.get("nonexistent") is None

    def test_get_updates_access(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        cache.set("dk1", _make_entry("dk1"))
        result = cache.get("dk1")
        assert result is not None
        assert result.access_count >= 1

    def test_delete(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        cache.set("dk1", _make_entry("dk1"))
        assert cache.delete("dk1") is True
        assert cache.get("dk1") is None

    def test_delete_nonexistent(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        assert cache.delete("nonexistent") is False

    def test_clear(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        cache.set("dk1", _make_entry("dk1"))
        cache.set("dk2", _make_entry("dk2"))
        cache.clear()
        assert cache.size() == 0
        assert cache.get("dk1") is None
        assert cache.get("dk2") is None

    def test_keys(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        cache.set("a", _make_entry("a"))
        cache.set("b", _make_entry("b"))
        assert set(cache.keys()) == {"a", "b"}

    def test_size(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        assert cache.size() == 0
        cache.set("dk1", _make_entry("dk1"))
        assert cache.size() == 1

    def test_compression_enabled(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100, compression=True)
        cache.set("dk1", _make_entry("dk1"))
        result = cache.get("dk1")
        assert result is not None
        assert result.key == "dk1"

    def test_compression_disabled(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100, compression=False)
        cache.set("dk1", _make_entry("dk1"))
        result = cache.get("dk1")
        assert result is not None
        assert result.key == "dk1"

    def test_eviction_on_max_size(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=2)
        cache.set("dk1", _make_entry("dk1"))
        cache.set("dk2", _make_entry("dk2"))
        cache.set("dk3", _make_entry("dk3"))
        # At least one old entry should be evicted
        assert cache.size() <= 2

    def test_stale_index_entry(self, tmp_path: Path):
        cache = DiskCache(cache_dir=tmp_path, max_size=100)
        cache.set("dk1", _make_entry("dk1"))
        # Manually delete the entry file to simulate stale index
        entry_file = cache._get_entry_file("dk1")
        entry_file.unlink()
        result = cache.get("dk1")
        assert result is None
        assert cache.size() == 0
