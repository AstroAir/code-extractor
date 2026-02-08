"""Tests for pysearch.indexing.cache.statistics module."""

from __future__ import annotations

import pytest

from pysearch.indexing.cache.statistics import CacheStatistics


class TestCacheStatistics:
    """Tests for CacheStatistics class."""

    def test_init(self):
        stats = CacheStatistics()
        assert stats.stats.hits == 0
        assert stats.stats.misses == 0
        assert stats.get_hit_rate() == 0.0

    def test_record_hit(self):
        stats = CacheStatistics()
        stats.record_hit()
        assert stats.stats.hits == 1
        assert stats.get_hit_rate() == pytest.approx(100.0)

    def test_record_miss(self):
        stats = CacheStatistics()
        stats.record_miss()
        assert stats.stats.misses == 1
        assert stats.get_hit_rate() == 0.0

    def test_record_eviction(self):
        stats = CacheStatistics()
        stats.record_eviction(3)
        assert stats.stats.evictions == 3
        stats.record_eviction(2)
        assert stats.stats.evictions == 5

    def test_record_eviction_default(self):
        stats = CacheStatistics()
        stats.record_eviction()
        assert stats.stats.evictions == 1

    def test_record_invalidation(self):
        stats = CacheStatistics()
        stats.record_invalidation(4)
        assert stats.stats.invalidations == 4
        stats.record_invalidation()
        assert stats.stats.invalidations == 5

    def test_update_access_time(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.update_access_time(0.5)
        assert stats.stats.average_access_time > 0

    def test_update_access_time_no_requests(self):
        stats = CacheStatistics()
        stats.update_access_time(0.5)
        # No requests means average stays 0
        assert stats.stats.average_access_time == 0.0

    def test_update_entry_count(self):
        stats = CacheStatistics()
        stats.update_entry_count(42)
        assert stats.stats.total_entries == 42

    def test_update_size(self):
        stats = CacheStatistics()
        stats.update_size(1024)
        assert stats.stats.total_size_bytes == 1024

    def test_add_size(self):
        stats = CacheStatistics()
        stats.add_size(100)
        stats.add_size(200)
        assert stats.stats.total_size_bytes == 300

    def test_subtract_size(self):
        stats = CacheStatistics()
        stats.add_size(500)
        stats.subtract_size(200)
        assert stats.stats.total_size_bytes == 300

    def test_subtract_size_clamps_to_zero(self):
        stats = CacheStatistics()
        stats.add_size(100)
        stats.subtract_size(999)
        assert stats.stats.total_size_bytes == 0

    def test_get_hit_rate(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.record_hit()
        stats.record_miss()
        rate = stats.get_hit_rate()
        assert rate == pytest.approx(200.0 / 3)

    def test_get_hit_rate_no_requests(self):
        stats = CacheStatistics()
        assert stats.get_hit_rate() == 0.0

    def test_get_stats_dict(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.record_miss()
        d = stats.get_stats_dict()
        assert isinstance(d, dict)
        assert "hits" in d
        assert "misses" in d
        assert "hit_rate" in d
        assert "evictions" in d
        assert "invalidations" in d
        assert "total_entries" in d
        assert "total_size_bytes" in d
        assert "average_access_time" in d
        assert d["hits"] == 1
        assert d["misses"] == 1

    def test_get_stats_dict_with_additional(self):
        stats = CacheStatistics()
        d = stats.get_stats_dict(additional_stats={"custom_key": "custom_value"})
        assert d["custom_key"] == "custom_value"

    def test_reset_stats(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.record_miss()
        stats.record_eviction(5)
        stats.add_size(1024)
        stats.reset_stats()
        assert stats.stats.hits == 0
        assert stats.stats.misses == 0
        assert stats.stats.evictions == 0
        assert stats.stats.total_size_bytes == 0
        assert stats.get_hit_rate() == 0.0

    def test_get_performance_summary(self):
        stats = CacheStatistics()
        stats.record_hit()
        stats.record_miss()
        stats.update_entry_count(10)
        stats.update_size(2 * 1024 * 1024)
        summary = stats.get_performance_summary()
        assert isinstance(summary, str)
        assert "2 requests" in summary
        assert "10 entries" in summary
