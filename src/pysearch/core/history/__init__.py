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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import SearchResult

# Import main classes for backward compatibility
from .history_analytics import AnalyticsManager
from .history_bookmarks import BookmarkFolder, BookmarkManager
from .history_core import SearchCategory, SearchHistory, SearchHistoryEntry
from .history_export import ExportFormat, HistoryExporter
from .history_sessions import SearchSession, SessionManager

# Re-export everything for backward compatibility
__all__ = [
    "SearchHistory",
    "SearchHistoryEntry",
    "SearchCategory",
    "BookmarkManager",
    "BookmarkFolder",
    "SessionManager",
    "SearchSession",
    "AnalyticsManager",
    "HistoryExporter",
    "ExportFormat",
    "extract_languages_from_results",
]


def extract_languages_from_results(result: SearchResult) -> set[str]:
    """Extract programming languages from search results based on file extensions.

    This is a shared utility used by both history_core and history_sessions to
    avoid code duplication.
    """
    languages: set[str] = set()
    _ext_map: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
    }

    for item in result.items[:20]:
        ext = item.file.suffix.lower()
        lang = _ext_map.get(ext)
        if lang:
            languages.add(lang)

    return languages
