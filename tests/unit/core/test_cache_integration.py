"""Tests for pysearch.core.integrations.cache_integration module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.integrations.cache_integration import CacheIntegrationManager
from pysearch.core.types import Query, SearchItem, SearchResult, SearchStats


def _make_result(tmp_path: Path | None = None) -> SearchResult:
    items = []
    if tmp_path:
        items = [SearchItem(file=tmp_path / "a.py", start_line=1, end_line=1, lines=["x"])]
    return SearchResult(
        items=items,
        stats=SearchStats(files_scanned=1, files_matched=1, items=len(items), elapsed_ms=1.0),
    )


class TestCacheIntegrationManager:
    """Tests for CacheIntegrationManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = CacheIntegrationManager(cfg)
        assert mgr._caching_enabled is False
        assert mgr.cache_manager is None
        assert mgr.cache_ttl == 300.0

    def test_is_caching_enabled_default(self):
        mgr = CacheIntegrationManager(SearchConfig())
        assert mgr.is_caching_enabled() is False

    def test_disable_caching_when_not_enabled(self):
        mgr = CacheIntegrationManager(SearchConfig())
        mgr.disable_caching()  # should not raise
        assert mgr.is_caching_enabled() is False

    def test_generate_cache_key(self):
        mgr = CacheIntegrationManager(SearchConfig())
        q1 = Query(pattern="test", use_regex=False, context=2)
        q2 = Query(pattern="test", use_regex=True, context=2)
        key1 = mgr._generate_cache_key(q1)
        key2 = mgr._generate_cache_key(q2)
        assert key1 != key2

    def test_generate_cache_key_same_query(self):
        mgr = CacheIntegrationManager(SearchConfig())
        q1 = Query(pattern="test", use_regex=False)
        q2 = Query(pattern="test", use_regex=False)
        assert mgr._generate_cache_key(q1) == mgr._generate_cache_key(q2)

    def test_get_file_dependencies(self, tmp_path: Path):
        mgr = CacheIntegrationManager(SearchConfig())
        result = _make_result(tmp_path)
        deps = mgr._get_file_dependencies(result)
        assert len(deps) == 1
        assert deps[0] == tmp_path / "a.py"

    def test_legacy_cache_result_and_retrieve(self):
        mgr = CacheIntegrationManager(SearchConfig())
        q = Query(pattern="cached_query")
        r = _make_result()
        mgr._cache_legacy_result(q, r)
        cached = mgr._get_legacy_cached_result(q)
        assert cached is not None
        assert cached is r

    def test_legacy_cache_expired(self):
        mgr = CacheIntegrationManager(SearchConfig())
        mgr.cache_ttl = 0.1  # 100ms TTL
        q = Query(pattern="expire_test")
        r = _make_result()
        mgr._cache_legacy_result(q, r)
        time.sleep(0.2)
        cached = mgr._get_legacy_cached_result(q)
        assert cached is None

    def test_legacy_cache_miss(self):
        mgr = CacheIntegrationManager(SearchConfig())
        q = Query(pattern="never_cached")
        assert mgr._get_legacy_cached_result(q) is None

    def test_is_cache_valid(self):
        mgr = CacheIntegrationManager(SearchConfig())
        mgr.cache_ttl = 300.0
        assert mgr._is_cache_valid(time.time()) is True
        assert mgr._is_cache_valid(time.time() - 400) is False

    def test_cache_result_without_caching_enabled(self):
        mgr = CacheIntegrationManager(SearchConfig())
        q = Query(pattern="test")
        r = _make_result()
        mgr.cache_result(q, r)
        # Should use legacy cache
        cached = mgr.get_cached_result(q)
        assert cached is not None

    def test_get_cached_result_without_caching(self):
        mgr = CacheIntegrationManager(SearchConfig())
        q = Query(pattern="test")
        r = _make_result()
        mgr._cache_legacy_result(q, r)
        cached = mgr.get_cached_result(q)
        assert cached is r

    def test_clear_caches(self):
        mgr = CacheIntegrationManager(SearchConfig())
        q = Query(pattern="clear_test")
        mgr._cache_legacy_result(q, _make_result())
        mgr.clear_caches()
        assert mgr._get_legacy_cached_result(q) is None
        assert len(mgr._file_content_cache) == 0
        assert len(mgr._search_result_cache) == 0

    def test_get_cache_stats(self):
        mgr = CacheIntegrationManager(SearchConfig())
        stats = mgr.get_cache_stats()
        assert stats["enabled"] is False
        assert stats["file_content_cache_size"] == 0
        assert stats["search_result_cache_size"] == 0

    def test_get_cached_file_content(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = CacheIntegrationManager(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')", encoding="utf-8")
        content = mgr.get_cached_file_content(test_file)
        assert content is not None
        assert "hello" in content

    def test_get_cached_file_content_caches(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = CacheIntegrationManager(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("cached_content", encoding="utf-8")
        mgr.get_cached_file_content(test_file)
        assert test_file in mgr._file_content_cache

    def test_get_cached_file_content_nonexistent(self, tmp_path: Path):
        mgr = CacheIntegrationManager(SearchConfig())
        result = mgr.get_cached_file_content(tmp_path / "nonexistent.py")
        assert result is None

    def test_invalidate_file_cache(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = CacheIntegrationManager(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("content", encoding="utf-8")
        mgr.get_cached_file_content(test_file)
        assert test_file in mgr._file_content_cache
        mgr.invalidate_file_cache(test_file)
        assert test_file not in mgr._file_content_cache

    def test_set_cache_ttl(self):
        mgr = CacheIntegrationManager(SearchConfig())
        mgr.set_cache_ttl(600.0)
        assert mgr.cache_ttl == 600.0

    def test_get_cache_hit_rate_no_manager(self):
        mgr = CacheIntegrationManager(SearchConfig())
        assert mgr.get_cache_hit_rate() == 0.0

    def test_legacy_cache_eviction(self):
        mgr = CacheIntegrationManager(SearchConfig())
        # Fill beyond 100 entries
        for i in range(120):
            q = Query(pattern=f"evict_{i}")
            mgr._cache_legacy_result(q, _make_result())
        assert len(mgr._search_result_cache) <= 100
