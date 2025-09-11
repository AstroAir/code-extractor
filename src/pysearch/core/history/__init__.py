"""
Search history tracking and management for pysearch.

This package provides comprehensive search history functionality organized into
focused modules:

- history_core: Basic history tracking and management
- history_bookmarks: Bookmark organization and management
- history_sessions: Search session tracking and analytics
- history_analytics: Search analytics and pattern analysis

For backward compatibility, the main SearchHistory class is re-exported from this package.
"""

# Import main classes for backward compatibility
from .history_core import SearchHistory, SearchHistoryEntry, SearchCategory

# Re-export everything for backward compatibility
__all__ = [
    "SearchHistory",
    "SearchHistoryEntry", 
    "SearchCategory",
]
