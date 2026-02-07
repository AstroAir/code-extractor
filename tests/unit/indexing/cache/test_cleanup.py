"""Tests for pysearch.indexing.cache.cleanup module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.indexing.cache.cleanup import CacheCleanup
from pysearch.indexing.cache.backends import MemoryCache
from pysearch.indexing.cache.models import CacheEntry
from pysearch.core.types import SearchResult, SearchStats


class TestCacheCleanup:
    """Tests for CacheCleanup class."""

    def test_init(self):
        backend = MemoryCache(max_size=100)
        cleanup = CacheCleanup(backend)
        assert cleanup is not None

    def test_is_running_default(self):
        backend = MemoryCache(max_size=100)
        cleanup = CacheCleanup(backend)
        # auto_cleanup may start thread, just check type
        assert isinstance(cleanup.is_running(), bool)

    def test_get_status(self):
        backend = MemoryCache(max_size=100)
        cleanup = CacheCleanup(backend)
        status = cleanup.get_status()
        assert isinstance(status, dict)

    def test_manual_cleanup(self):
        backend = MemoryCache(max_size=100)
        cleanup = CacheCleanup(backend)
        result = cleanup.manual_cleanup()
        assert isinstance(result, int)
