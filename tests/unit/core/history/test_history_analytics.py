"""Tests for pysearch.core.history.history_analytics module."""

from __future__ import annotations

import time

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.history.history_analytics import AnalyticsManager
from pysearch.core.history.history_core import SearchCategory, SearchHistoryEntry


def _make_entry(
    pattern: str = "test",
    category: SearchCategory = SearchCategory.GENERAL,
    items_count: int = 5,
    elapsed_ms: float = 50.0,
    success_score: float = 0.7,
    use_regex: bool = False,
    use_ast: bool = False,
    languages: set[str] | None = None,
    tags: set[str] | None = None,
    user_rating: int | None = None,
    timestamp: float | None = None,
) -> SearchHistoryEntry:
    return SearchHistoryEntry(
        timestamp=timestamp or time.time(),
        query_pattern=pattern,
        use_regex=use_regex,
        use_ast=use_ast,
        context=2,
        files_matched=2,
        items_count=items_count,
        elapsed_ms=elapsed_ms,
        category=category,
        success_score=success_score,
        languages=languages,
        tags=tags,
        user_rating=user_rating,
    )


class TestAnalyticsManager:
    """Tests for AnalyticsManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = AnalyticsManager(cfg)
        assert mgr.cfg is cfg

    def test_get_search_analytics_empty(self):
        mgr = AnalyticsManager(SearchConfig())
        result = mgr.get_search_analytics([], days=30)
        assert result["total_searches"] == 0
        assert result["success_rate"] == 0.0

    def test_get_search_analytics_with_entries(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("def foo", SearchCategory.FUNCTION, items_count=5, elapsed_ms=100.0),
            _make_entry("class Bar", SearchCategory.CLASS, items_count=3, elapsed_ms=150.0),
            _make_entry("import os", SearchCategory.IMPORT, items_count=1, elapsed_ms=50.0),
            _make_entry("nothing", SearchCategory.GENERAL, items_count=0, elapsed_ms=200.0),
        ]
        result = mgr.get_search_analytics(entries, days=1)
        assert result["total_searches"] == 4
        assert result["successful_searches"] == 3
        assert result["success_rate"] == 0.75
        assert result["average_search_time"] == 125.0

    def test_get_search_analytics_categories(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry(category=SearchCategory.FUNCTION),
            _make_entry(category=SearchCategory.FUNCTION),
            _make_entry(category=SearchCategory.CLASS),
        ]
        result = mgr.get_search_analytics(entries, days=1)
        categories = dict(result["most_common_categories"])
        assert categories["function"] == 2
        assert categories["class"] == 1

    def test_get_search_analytics_languages(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry(languages={"python", "javascript"}),
            _make_entry(languages={"python"}),
        ]
        result = mgr.get_search_analytics(entries, days=1)
        langs = dict(result["most_used_languages"])
        assert langs["python"] == 2
        assert langs["javascript"] == 1

    def test_get_search_analytics_old_entries_excluded(self):
        mgr = AnalyticsManager(SearchConfig())
        old_time = time.time() - 100 * 86400  # 100 days ago
        entries = [
            _make_entry(timestamp=old_time),
        ]
        result = mgr.get_search_analytics(entries, days=30)
        assert result["total_searches"] == 0

    def test_get_pattern_suggestions(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("process_data"),
            _make_entry("process_info"),
            _make_entry("handle_request"),
        ]
        suggestions = mgr.get_pattern_suggestions(entries, "process", limit=5)
        assert len(suggestions) == 2
        assert all("process" in s for s in suggestions)

    def test_get_pattern_suggestions_limit(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry(f"test_{i}") for i in range(10)]
        suggestions = mgr.get_pattern_suggestions(entries, "test", limit=3)
        assert len(suggestions) == 3

    def test_get_pattern_suggestions_no_match(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("foo")]
        suggestions = mgr.get_pattern_suggestions(entries, "xyz", limit=5)
        assert suggestions == []

    def test_get_performance_insights_empty(self):
        mgr = AnalyticsManager(SearchConfig())
        result = mgr.get_performance_insights([])
        assert result["insights"] == []
        assert result["recommendations"] == []

    def test_get_performance_insights_slow_searches(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry(elapsed_ms=50.0),
            _make_entry(elapsed_ms=50.0),
            _make_entry(elapsed_ms=500.0),  # slow
        ]
        result = mgr.get_performance_insights(entries)
        assert any("slow" in i.lower() for i in result["insights"])

    def test_get_performance_insights_empty_results(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry(items_count=0, elapsed_ms=10.0),
            _make_entry(items_count=0, elapsed_ms=10.0),
            _make_entry(items_count=0, elapsed_ms=10.0),
            _make_entry(items_count=5, elapsed_ms=10.0),
        ]
        result = mgr.get_performance_insights(entries)
        assert any("no results" in i.lower() for i in result["insights"])

    def test_get_usage_patterns_empty(self):
        mgr = AnalyticsManager(SearchConfig())
        assert mgr.get_usage_patterns([]) == {}

    def test_get_usage_patterns_with_data(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("short"),
            _make_entry("a_longer_pattern_here", use_regex=True),
        ]
        result = mgr.get_usage_patterns(entries)
        assert "temporal_patterns" in result
        assert "search_patterns" in result
        assert "productivity_metrics" in result
        assert result["search_patterns"]["regex_usage_rate"] == 0.5

    def test_rate_search(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("foo"), _make_entry("bar")]
        assert mgr.rate_search(entries, "bar", 5) is True
        assert entries[1].user_rating == 5

    def test_rate_search_invalid_rating(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("foo")]
        assert mgr.rate_search(entries, "foo", 0) is False
        assert mgr.rate_search(entries, "foo", 6) is False

    def test_rate_search_not_found(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("foo")]
        assert mgr.rate_search(entries, "nonexistent", 3) is False

    def test_add_tags_to_search(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("foo")]
        assert mgr.add_tags_to_search(entries, "foo", {"tag1", "tag2"}) is True
        assert entries[0].tags == {"tag1", "tag2"}

    def test_add_tags_to_search_append(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("foo", tags={"existing"})]
        mgr.add_tags_to_search(entries, "foo", {"new"})
        assert entries[0].tags == {"existing", "new"}

    def test_add_tags_not_found(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("foo")]
        assert mgr.add_tags_to_search(entries, "bar", {"tag"}) is False

    def test_search_history_by_tags(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("a", tags={"python", "api"}),
            _make_entry("b", tags={"javascript"}),
            _make_entry("c"),
        ]
        result = mgr.search_history_by_tags(entries, {"python"})
        assert len(result) == 1
        assert result[0].query_pattern == "a"

    def test_search_history_by_tags_intersection(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("a", tags={"python", "api"}),
            _make_entry("b", tags={"api", "rest"}),
        ]
        result = mgr.search_history_by_tags(entries, {"api"})
        assert len(result) == 2

    def test_search_history_by_tags_no_match(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [_make_entry("a", tags={"python"})]
        result = mgr.search_history_by_tags(entries, {"nonexistent"})
        assert result == []


class TestAnalyticsTrends:
    """Tests for trend analysis methods."""

    def test_get_search_trends_empty(self):
        mgr = AnalyticsManager(SearchConfig())
        result = mgr.get_search_trends([], days=30)
        assert result["daily_counts"] == {}

    def test_get_search_trends_with_data(self):
        mgr = AnalyticsManager(SearchConfig())
        now = time.time()
        entries = [
            _make_entry("a", items_count=5, elapsed_ms=100.0, timestamp=now - 86400),
            _make_entry("b", items_count=0, elapsed_ms=50.0, timestamp=now - 86400),
            _make_entry("c", items_count=3, elapsed_ms=200.0, timestamp=now),
        ]
        result = mgr.get_search_trends(entries, days=7)

        assert len(result["daily_counts"]) >= 1
        assert result["total_days_active"] >= 1
        assert result["peak_count"] >= 1
        assert result["peak_day"] is not None
        assert "trend" in result

    def test_get_search_trends_success_rates(self):
        mgr = AnalyticsManager(SearchConfig())
        now = time.time()
        entries = [
            _make_entry("a", items_count=5, timestamp=now),
            _make_entry("b", items_count=0, timestamp=now),
        ]
        result = mgr.get_search_trends(entries, days=1)

        assert len(result["daily_success_rates"]) == 1
        day = list(result["daily_success_rates"].keys())[0]
        assert result["daily_success_rates"][day] == 0.5

    def test_get_search_trends_excludes_old(self):
        mgr = AnalyticsManager(SearchConfig())
        old_time = time.time() - 100 * 86400
        entries = [_make_entry("old", timestamp=old_time)]
        result = mgr.get_search_trends(entries, days=30)
        assert result["daily_counts"] == {}

    def test_get_category_trends_empty(self):
        mgr = AnalyticsManager(SearchConfig())
        result = mgr.get_category_trends([], days=30)
        assert result["weekly_categories"] == {}
        assert result["category_shifts"] == []

    def test_get_category_trends_with_data(self):
        mgr = AnalyticsManager(SearchConfig())
        now = time.time()
        entries = [
            _make_entry(category=SearchCategory.FUNCTION, timestamp=now),
            _make_entry(category=SearchCategory.CLASS, timestamp=now),
            _make_entry(category=SearchCategory.FUNCTION, timestamp=now - 86400),
        ]
        result = mgr.get_category_trends(entries, days=7)

        assert result["weeks_analyzed"] >= 1
        assert len(result["weekly_categories"]) >= 1

    def test_get_top_failed_patterns_empty(self):
        mgr = AnalyticsManager(SearchConfig())
        result = mgr.get_top_failed_patterns([], limit=10)
        assert result == []

    def test_get_top_failed_patterns_with_failures(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("good", items_count=5),
            _make_entry("bad1", items_count=0),
            _make_entry("bad1", items_count=0),
            _make_entry("bad2", items_count=0),
            _make_entry("bad2", items_count=3),
        ]
        result = mgr.get_top_failed_patterns(entries, limit=10)

        assert len(result) >= 2
        # bad1 should be first (2 failures)
        assert result[0]["pattern"] == "bad1"
        assert result[0]["failed_searches"] == 2
        assert result[0]["failure_rate"] == 1.0

    def test_get_top_failed_patterns_with_limit(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry(f"fail_{i}", items_count=0)
            for i in range(20)
        ]
        result = mgr.get_top_failed_patterns(entries, limit=5)
        assert len(result) == 5

    def test_get_top_failed_patterns_no_failures(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("success1", items_count=5),
            _make_entry("success2", items_count=3),
        ]
        result = mgr.get_top_failed_patterns(entries, limit=10)
        assert result == []

    def test_get_top_failed_patterns_mixed(self):
        mgr = AnalyticsManager(SearchConfig())
        entries = [
            _make_entry("mixed", items_count=5),
            _make_entry("mixed", items_count=0),
            _make_entry("mixed", items_count=3),
        ]
        result = mgr.get_top_failed_patterns(entries, limit=10)

        assert len(result) == 1
        assert result[0]["pattern"] == "mixed"
        assert result[0]["total_searches"] == 3
        assert result[0]["failed_searches"] == 1
        assert abs(result[0]["failure_rate"] - 1 / 3) < 0.01
