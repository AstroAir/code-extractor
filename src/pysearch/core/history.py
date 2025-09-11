"""
Search history tracking and management for pysearch.

This module provides backward compatibility by re-exporting the main SearchHistory
class from the new modular structure. All history functionality is now organized
in the history/ package but can still be imported from this module for compatibility.

For new code, consider importing directly from the specific history modules:
- from pysearch.core.history.history_core import SearchHistory
- from pysearch.core.history.history_bookmarks import BookmarkManager
- from pysearch.core.history.history_sessions import SessionManager
- from pysearch.core.history.history_analytics import AnalyticsManager

Example:
    Basic history usage:
        >>> from pysearch.core.history import SearchHistory
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> history = SearchHistory(config)
        >>>
        >>> # Record a search
        >>> history.add_search(query, results)
        >>>
        >>> # Get recent searches
        >>> recent = history.get_history(limit=10)
        >>> for entry in recent:
        ...     print(f"{entry.query_pattern}: {entry.items_count} results")
"""

# Import all classes from the new modular structure for backward compatibility
from .history import *  # noqa: F403, F401
