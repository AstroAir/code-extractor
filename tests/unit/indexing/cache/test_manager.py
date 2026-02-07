"""Tests for pysearch.indexing.cache.manager module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.types import SearchResult, SearchStats
from pysearch.indexing.cache.manager import CacheManager


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init_memory(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        assert mgr is not None

    def test_init_disk(self, tmp_path: Path):
        mgr = CacheManager(backend="disk", cache_dir=tmp_path, max_size=100, auto_cleanup=False)
        assert mgr is not None

    def test_set_and_get(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = SearchResult(items=[], stats=SearchStats(files_scanned=5))
        mgr.set("test_query", result)
        cached = mgr.get("test_query")
        assert cached is not None

    def test_get_nonexistent(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        assert mgr.get("nonexistent") is None

    def test_delete(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = SearchResult(items=[], stats=SearchStats())
        mgr.set("k1", result)
        mgr.delete("k1")
        assert mgr.get("k1") is None

    def test_invalidate_by_file(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = SearchResult(items=[], stats=SearchStats())
        mgr.set("k1", result, file_dependencies={"a.py"})
        mgr.invalidate_by_file("a.py")
        assert mgr.get("k1") is None

    def test_clear(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = SearchResult(items=[], stats=SearchStats())
        mgr.set("k1", result)
        mgr.set("k2", result)
        mgr.clear()
        assert mgr.get("k1") is None
        assert mgr.get("k2") is None

    def test_get_stats(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        stats = mgr.get_stats()
        assert isinstance(stats, dict)
