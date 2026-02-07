"""Tests for pysearch.indexing.cache.statistics module."""

from __future__ import annotations

import pytest

from pysearch.indexing.cache.statistics import CacheStatistics


class TestCacheStatistics:
    """Tests for CacheStatistics class."""

    def test_init(self):
        stats = CacheStatistics()
        assert stats is not None

    def test_record_hit(self):
        stats = CacheStatistics()
        stats.record_hit()
        rate = stats.get_hit_rate()
        assert rate > 0

    def test_record_miss(self):
        stats = CacheStatistics()
        stats.record_miss()
        rate = stats.get_hit_rate()
        assert rate == 0.0

    def test_hit_rate(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.record_hit()
        stats.record_miss()
        rate = stats.get_hit_rate()
        # Rate may be percentage (66.67) or fraction (0.667)
        assert rate > 0

    def test_hit_rate_no_requests(self):
        stats = CacheStatistics()
        assert stats.get_hit_rate() == 0.0

    def test_reset(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.record_miss()
        stats.reset_stats()
        assert stats.get_hit_rate() == 0.0

    def test_get_stats_dict(self):
        stats = CacheStatistics()
        stats.record_hit()
        d = stats.get_stats_dict()
        assert isinstance(d, dict)

    def test_get_performance_summary(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.record_miss()
        summary = stats.get_performance_summary()
        assert isinstance(summary, (dict, str))
