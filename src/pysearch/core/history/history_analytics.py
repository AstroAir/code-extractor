"""
Search analytics and pattern analysis functionality.

This module provides comprehensive analytics capabilities for search history,
including pattern analysis, success metrics, and search behavior insights.

Classes:
    AnalyticsManager: Main analytics and reporting class

Key Features:
    - Comprehensive search analytics and metrics
    - Pattern suggestion based on history
    - Search behavior analysis
    - Performance insights and recommendations

Example:
    Analytics usage:
        >>> from pysearch.core.history.history_analytics import AnalyticsManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = AnalyticsManager(config)
        >>>
        >>> # Get analytics for recent searches
        >>> analytics = manager.get_search_analytics(history_entries, days=30)
        >>> print(f"Success rate: {analytics['success_rate']:.2%}")
        >>> print(f"Average search time: {analytics['average_search_time']:.1f}ms")
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from ..config import SearchConfig
from .history_core import SearchHistoryEntry


class AnalyticsManager:
    """Search analytics and pattern analysis."""

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg

    def get_search_analytics(
        self, history_entries: list[SearchHistoryEntry], days: int = 30
    ) -> dict[str, Any]:
        """Get comprehensive search analytics."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        recent_entries = [entry for entry in history_entries if entry.timestamp >= cutoff_time]

        if not recent_entries:
            return {
                "total_searches": 0,
                "successful_searches": 0,
                "average_success_score": 0.0,
                "most_common_categories": [],
                "most_used_languages": [],
                "average_search_time": 0.0,
                "search_frequency": {},
                "success_rate": 0.0,
            }

        # Calculate statistics
        total_searches = len(recent_entries)
        successful_searches = sum(1 for entry in recent_entries if entry.items_count > 0)
        average_success_score = (
            sum(entry.success_score for entry in recent_entries) / total_searches
        )
        average_search_time = sum(entry.elapsed_ms for entry in recent_entries) / total_searches

        # Category analysis
        category_counts: dict[str, int] = defaultdict(int)
        for entry in recent_entries:
            category_counts[entry.category.value] += 1
        most_common_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        # Language analysis
        language_counts: dict[str, int] = defaultdict(int)
        for entry in recent_entries:
            if entry.languages:
                for lang in entry.languages:
                    language_counts[lang] += 1
        most_used_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Search frequency by day
        search_frequency: dict[str, int] = defaultdict(int)
        for entry in recent_entries:
            day = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            search_frequency[day] += 1

        # Count unique sessions (approximate from entry timestamps with 30-min gaps)
        session_count = 1
        sorted_entries = sorted(recent_entries, key=lambda e: e.timestamp)
        for i in range(1, len(sorted_entries)):
            if sorted_entries[i].timestamp - sorted_entries[i - 1].timestamp > 1800:
                session_count += 1

        return {
            "total_searches": total_searches,
            "successful_searches": successful_searches,
            "success_rate": successful_searches / total_searches if total_searches > 0 else 0,
            "average_success_score": average_success_score,
            "most_common_categories": most_common_categories,
            "most_used_languages": most_used_languages,
            "average_search_time": average_search_time,
            "search_frequency": dict(search_frequency),
            "session_count": session_count,
        }

    def get_pattern_suggestions(
        self, history_entries: list[SearchHistoryEntry], partial_pattern: str, limit: int = 5
    ) -> list[str]:
        """Get pattern suggestions based on search history."""
        partial_lower = partial_pattern.lower()

        suggestions = []
        seen = set()

        # Look for patterns that start with or contain the partial pattern
        for entry in reversed(history_entries):  # Most recent first
            pattern = entry.query_pattern
            if (
                pattern.lower().startswith(partial_lower) or partial_lower in pattern.lower()
            ) and pattern not in seen:
                suggestions.append(pattern)
                seen.add(pattern)

                if len(suggestions) >= limit:
                    break

        return suggestions

    def get_performance_insights(self, history_entries: list[SearchHistoryEntry]) -> dict[str, Any]:
        """Get performance insights and recommendations."""
        if not history_entries:
            return {"insights": [], "recommendations": []}

        insights = []
        recommendations = []

        # Analyze search times
        search_times = [entry.elapsed_ms for entry in history_entries]
        avg_time = sum(search_times) / len(search_times)
        slow_searches = [entry for entry in history_entries if entry.elapsed_ms > avg_time * 2]

        if slow_searches:
            insights.append(f"Found {len(slow_searches)} slow searches (>{avg_time*2:.1f}ms)")
            recommendations.append(
                "Consider using more specific patterns or filters for better performance"
            )

        # Analyze success rates by category
        category_success: dict[str, list[float]] = defaultdict(list)
        for entry in history_entries:
            category_success[entry.category.value].append(entry.success_score)

        low_success_categories = []
        for category, scores in category_success.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.5 and len(scores) >= 3:  # At least 3 searches
                low_success_categories.append((category, avg_score))

        if low_success_categories:
            insights.append(
                f"Low success rate in categories: {', '.join(cat for cat, _ in low_success_categories)}"
            )
            recommendations.append(
                "Try using AST filters or more specific patterns for better results"
            )

        # Analyze pattern complexity
        complex_patterns = [
            entry for entry in history_entries if len(entry.query_pattern) > 50 or entry.use_regex
        ]
        if len(complex_patterns) > len(history_entries) * 0.3:  # More than 30% complex
            insights.append("High usage of complex patterns detected")
            recommendations.append("Consider breaking complex searches into simpler ones")

        # Analyze result counts
        empty_results = [entry for entry in history_entries if entry.items_count == 0]
        if len(empty_results) > len(history_entries) * 0.2:  # More than 20% empty
            insights.append(f"{len(empty_results)} searches returned no results")
            recommendations.append("Try broader patterns or check file paths and filters")

        return {
            "insights": insights,
            "recommendations": recommendations,
            "metrics": {
                "average_search_time": avg_time,
                "slow_search_count": len(slow_searches),
                "empty_result_rate": len(empty_results) / len(history_entries),
                "complex_pattern_rate": len(complex_patterns) / len(history_entries),
            },
        }

    def get_usage_patterns(self, history_entries: list[SearchHistoryEntry]) -> dict[str, Any]:
        """Analyze usage patterns and trends."""
        if not history_entries:
            return {}

        # Time-based patterns
        hour_counts: dict[int, int] = defaultdict(int)
        day_counts: dict[int, int] = defaultdict(int)  # 0=Monday, 6=Sunday

        for entry in history_entries:
            dt = datetime.fromtimestamp(entry.timestamp)
            hour_counts[dt.hour] += 1
            day_counts[dt.weekday()] += 1

        # Most active hours and days
        most_active_hour = max(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (0, 0)
        most_active_day = max(day_counts.items(), key=lambda x: x[1]) if day_counts else (0, 0)

        # Pattern length analysis
        pattern_lengths = [len(entry.query_pattern) for entry in history_entries]
        avg_pattern_length = sum(pattern_lengths) / len(pattern_lengths)

        # Search mode preferences
        regex_usage = sum(1 for entry in history_entries if entry.use_regex)
        ast_usage = sum(1 for entry in history_entries if entry.use_ast)

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return {
            "temporal_patterns": {
                "most_active_hour": most_active_hour[0],
                "most_active_day": day_names[most_active_day[0]],
                "hourly_distribution": dict(hour_counts),
                "daily_distribution": {day_names[k]: v for k, v in day_counts.items()},
            },
            "search_patterns": {
                "average_pattern_length": avg_pattern_length,
                "regex_usage_rate": regex_usage / len(history_entries),
                "ast_usage_rate": ast_usage / len(history_entries),
            },
            "productivity_metrics": {
                "searches_per_day": len(history_entries)
                / max(1, (time.time() - min(e.timestamp for e in history_entries)) / 86400),
                "average_results_per_search": sum(e.items_count for e in history_entries)
                / len(history_entries),
            },
        }

    def rate_search(
        self, history_entries: list[SearchHistoryEntry], pattern: str, rating: int
    ) -> bool:
        """Rate a search result (1-5 stars)."""
        if not 1 <= rating <= 5:
            return False

        # Find the most recent search with this pattern
        for entry in reversed(history_entries):
            if entry.query_pattern == pattern:
                entry.user_rating = rating
                return True

        return False

    def add_tags_to_search(
        self, history_entries: list[SearchHistoryEntry], pattern: str, tags: set[str]
    ) -> bool:
        """Add tags to a search in history."""
        # Find the most recent search with this pattern
        for entry in reversed(history_entries):
            if entry.query_pattern == pattern:
                if entry.tags:
                    entry.tags.update(tags)
                else:
                    entry.tags = tags.copy()
                return True

        return False

    def search_history_by_tags(
        self, history_entries: list[SearchHistoryEntry], tags: set[str]
    ) -> list[SearchHistoryEntry]:
        """Find searches by tags."""
        matching_entries = []
        for entry in history_entries:
            if entry.tags and tags.intersection(entry.tags):
                matching_entries.append(entry)

        return sorted(matching_entries, key=lambda e: e.timestamp, reverse=True)
