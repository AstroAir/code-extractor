"""Tests for pysearch.indexing.cache.manager module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.core.types import SearchResult, SearchStats
from pysearch.indexing.cache.manager import CacheManager


def _make_result(**kwargs) -> SearchResult:
    return SearchResult(items=[], stats=SearchStats(**kwargs))


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init_memory(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        assert mgr.backend is not None
        assert mgr.default_ttl == 3600

    def test_init_disk(self, tmp_path: Path):
        mgr = CacheManager(
            backend="disk", cache_dir=tmp_path, max_size=100, auto_cleanup=False
        )
        assert mgr.backend is not None

    def test_init_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown cache backend"):
            CacheManager(backend="redis", auto_cleanup=False)

    def test_set_and_get(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result(files_scanned=5)
        assert mgr.set("test_query", result) is True
        cached = mgr.get("test_query")
        assert cached is not None

    def test_get_nonexistent(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        assert mgr.get("nonexistent") is None

    def test_get_expired_entry(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result, ttl=0.01)
        time.sleep(0.05)
        assert mgr.get("k1") is None

    def test_set_with_custom_ttl(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result, ttl=7200)
        cached = mgr.get("k1")
        assert cached is not None

    def test_set_with_file_dependencies(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result, file_dependencies={"a.py", "b.py"})
        deps = mgr.dependencies.get_dependent_keys("a.py")
        assert "k1" in deps

    def test_delete(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result)
        assert mgr.delete("k1") is True
        assert mgr.get("k1") is None

    def test_delete_nonexistent(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        assert mgr.delete("nonexistent") is False

    def test_invalidate_by_file(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result, file_dependencies={"a.py"})
        mgr.set("k2", result, file_dependencies={"a.py"})
        mgr.set("k3", result, file_dependencies={"b.py"})
        count = mgr.invalidate_by_file("a.py")
        assert count == 2
        assert mgr.get("k1") is None
        assert mgr.get("k2") is None
        assert mgr.get("k3") is not None

    def test_invalidate_by_file_no_match(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        count = mgr.invalidate_by_file("nonexistent.py")
        assert count == 0

    def test_invalidate_by_pattern(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("search:foo", result)
        mgr.set("search:bar", result)
        mgr.set("other:baz", result)
        count = mgr.invalidate_by_pattern("search:*")
        assert count == 2
        assert mgr.get("search:foo") is None
        assert mgr.get("search:bar") is None
        assert mgr.get("other:baz") is not None

    def test_invalidate_by_pattern_no_match(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result)
        count = mgr.invalidate_by_pattern("nomatch:*")
        assert count == 0

    def test_clear(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result)
        mgr.set("k2", result)
        mgr.clear()
        assert mgr.get("k1") is None
        assert mgr.get("k2") is None

    def test_cleanup_expired(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("expired", result, ttl=0.01)
        mgr.set("valid", result, ttl=3600)
        time.sleep(0.05)
        removed = mgr.cleanup_expired()
        assert removed >= 1
        assert mgr.get("expired") is None
        assert mgr.get("valid") is not None

    def test_get_stats(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        result = _make_result()
        mgr.set("k1", result)
        mgr.get("k1")
        mgr.get("nonexistent")
        stats = mgr.get_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "total_entries" in stats
        assert "file_dependencies" in stats
        assert "performance_summary" in stats

    def test_set_default_ttl(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=False)
        assert mgr.default_ttl == 3600
        mgr.set_default_ttl(1800)
        assert mgr.default_ttl == 1800

    def test_shutdown(self):
        mgr = CacheManager(backend="memory", max_size=100, auto_cleanup=True)
        mgr.shutdown()
        assert mgr.cleanup_manager.is_running() is False

    def test_context_manager(self):
        with CacheManager(backend="memory", max_size=100, auto_cleanup=False) as mgr:
            result = _make_result()
            mgr.set("k1", result)
            assert mgr.get("k1") is not None
        # After exit, shutdown should have been called
        assert mgr.cleanup_manager.is_running() is False
