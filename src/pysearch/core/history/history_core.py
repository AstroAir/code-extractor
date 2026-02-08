"""
Core search history tracking functionality.

This module provides the fundamental search history tracking capabilities,
including the main SearchHistory class and basic history management operations.

Classes:
    SearchCategory: Enumeration of search categories
    SearchHistoryEntry: Individual search record
    SearchHistory: Main history management class

Key Features:
    - Track all search operations with detailed metadata
    - Categorize searches by type (function, class, variable, etc.)
    - Persistent storage of search history
    - Basic search history retrieval and management

Example:
    Basic history usage:
        >>> from pysearch.core.history.history_core import SearchHistory
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

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from ..config import SearchConfig
from ..types import Query, SearchResult


class SearchCategory(str, Enum):
    """Categories for organizing searches."""

    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    COMMENT = "comment"
    STRING = "string"
    REGEX = "regex"
    GENERAL = "general"


@dataclass(slots=True)
class SearchHistoryEntry:
    timestamp: float
    query_pattern: str
    use_regex: bool
    use_ast: bool
    context: int
    files_matched: int
    items_count: int
    elapsed_ms: float
    filters: str | None = None
    # Enhanced fields
    session_id: str | None = None
    category: SearchCategory = SearchCategory.GENERAL
    languages: set[str] | None = None
    paths: list[str] | None = None
    success_score: float = 0.0  # 0-1 based on results quality
    user_rating: int | None = None  # 1-5 user rating
    tags: set[str] | None = None


class SearchHistory:
    """Core search history manager with basic tracking functionality."""

    def __init__(self, cfg: SearchConfig, max_entries: int = 1000) -> None:
        self.cfg = cfg
        self.max_entries = max_entries
        self.cache_dir = cfg.resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.cache_dir / "search_history.json"

        self._history: deque[SearchHistoryEntry] = deque(maxlen=max_entries)
        self._loaded = False

        # Import managers for extended functionality
        self._bookmark_manager: Any = None
        self._session_manager: Any = None
        self._analytics_manager: Any = None

    def _ensure_managers_loaded(self) -> None:
        """Lazy load the additional managers to avoid circular imports."""
        if self._bookmark_manager is None:
            from .history_bookmarks import BookmarkManager

            self._bookmark_manager = BookmarkManager(self.cfg)

        if self._session_manager is None:
            from .history_sessions import SessionManager

            self._session_manager = SessionManager(self.cfg)

        if self._analytics_manager is None:
            from .history_analytics import AnalyticsManager

            self._analytics_manager = AnalyticsManager(self.cfg)

    def load(self) -> None:
        """Load search history from disk."""
        if self._loaded:
            return

        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                entries = data.get("entries", [])
                # Keep only recent entries
                for entry_data in entries[-self.max_entries :]:
                    # Handle legacy entries without new fields
                    if "category" not in entry_data:
                        entry_data["category"] = SearchCategory.GENERAL
                    elif isinstance(entry_data["category"], str):
                        try:
                            entry_data["category"] = SearchCategory(entry_data["category"])
                        except ValueError:
                            entry_data["category"] = SearchCategory.GENERAL
                    if "languages" in entry_data and entry_data["languages"]:
                        entry_data["languages"] = set(entry_data["languages"])
                    if "tags" in entry_data and entry_data["tags"]:
                        entry_data["tags"] = set(entry_data["tags"])

                    entry = SearchHistoryEntry(**entry_data)
                    self._history.append(entry)
            except Exception:
                pass

        self._loaded = True

    def add_search(self, query: Query, result: SearchResult) -> None:
        """Add a search to history with enhanced tracking."""
        self.load()
        self._ensure_managers_loaded()
        current_time = time.time()

        # Get session from session manager
        if self._session_manager:
            session = self._session_manager.get_or_create_session(current_time)
        else:
            session = None

        # Categorize the search
        category = self._categorize_search(query)

        # Calculate success score based on results
        success_score = self._calculate_success_score(result)

        # Extract languages from results
        languages = self._extract_languages_from_results(result)

        # Extract paths
        paths = [str(item.file.parent) for item in result.items[:10]]  # Top 10 paths

        entry = SearchHistoryEntry(
            timestamp=current_time,
            query_pattern=query.pattern,
            use_regex=query.use_regex,
            use_ast=query.use_ast,
            context=query.context,
            files_matched=result.stats.files_matched,
            items_count=result.stats.items,
            elapsed_ms=result.stats.elapsed_ms,
            filters=str(query.filters) if query.filters else None,
            session_id=session.session_id if session else None,
            category=category,
            languages=languages,
            paths=paths,
            success_score=success_score,
        )

        self._history.append(entry)
        if self._session_manager and session:
            self._session_manager.update_session(session, query, result)

        self.save_history()
        if self._session_manager:
            self._session_manager.save_sessions()

    def _categorize_search(self, query: Query) -> SearchCategory:
        """Automatically categorize a search based on the query."""
        pattern = query.pattern.lower()

        # Function patterns
        if any(keyword in pattern for keyword in ["def ", "function", "func ", "()", "method"]):
            return SearchCategory.FUNCTION

        # Class patterns
        if any(keyword in pattern for keyword in ["class ", "interface", "struct", "enum"]):
            return SearchCategory.CLASS

        # Import patterns
        if any(keyword in pattern for keyword in ["import", "require", "include", "from "]):
            return SearchCategory.IMPORT

        # Variable patterns (common naming conventions)
        if (
            "_" in pattern
            or pattern.isupper()
            or any(keyword in pattern for keyword in ["var ", "let ", "const ", "="])
        ):
            return SearchCategory.VARIABLE

        # Comment patterns
        if pattern.startswith("#") or pattern.startswith("//") or pattern.startswith("/*"):
            return SearchCategory.COMMENT

        # String patterns
        if (pattern.startswith('"') and pattern.endswith('"')) or (
            pattern.startswith("'") and pattern.endswith("'")
        ):
            return SearchCategory.STRING

        # Regex patterns
        if query.use_regex or any(
            char in pattern for char in ["^", "$", "[", "]", "*", "+", "?", "|"]
        ):
            return SearchCategory.REGEX

        return SearchCategory.GENERAL

    def _calculate_success_score(self, result: SearchResult) -> float:
        """Calculate a success score based on search results quality."""
        if result.stats.items == 0:
            return 0.0

        # Base score from having results
        score = 0.3

        # Bonus for reasonable number of results (not too few, not too many)
        if 1 <= result.stats.items <= 20:
            score += 0.4
        elif 21 <= result.stats.items <= 100:
            score += 0.2
        elif result.stats.items > 100:
            score += 0.1

        # Bonus for multiple files (indicates broad relevance)
        if result.stats.files_matched > 1:
            score += 0.2

        # Bonus for reasonable search time (fast searches are often more targeted)
        if result.stats.elapsed_ms < 100:
            score += 0.1

        return min(1.0, score)

    def _extract_languages_from_results(self, result: SearchResult) -> set[str]:
        """Extract programming languages from search results."""
        from . import extract_languages_from_results

        return extract_languages_from_results(result)

    def save_history(self) -> None:
        """Save search history to disk."""
        try:
            entries_data = []
            for entry in self._history:
                entry_dict = asdict(entry)
                # Convert sets to lists for JSON serialization
                if isinstance(entry_dict.get("languages"), set):
                    entry_dict["languages"] = list(entry_dict["languages"])
                if isinstance(entry_dict.get("tags"), set):
                    entry_dict["tags"] = list(entry_dict["tags"])
                # Ensure category is serialized as string value
                if hasattr(entry_dict.get("category"), "value"):
                    entry_dict["category"] = entry_dict["category"].value
                entries_data.append(entry_dict)

            data = {
                "version": 1,
                "last_updated": time.time(),
                "entries": entries_data,
            }
            tmp_file = self.history_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.history_file)
        except Exception:
            pass

    def get_history(self, limit: int | None = None) -> list[SearchHistoryEntry]:
        """Get search history, most recent first."""
        self.load()
        history_list = list(reversed(self._history))
        if limit:
            return history_list[:limit]
        return history_list

    def get_frequent_patterns(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most frequently used search patterns."""
        self.load()
        pattern_counts: dict[str, int] = {}

        for entry in self._history:
            pattern = entry.query_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_patterns[:limit]

    def get_recent_patterns(self, days: int = 7, limit: int = 20) -> list[str]:
        """Get recently used patterns within specified days."""
        self.load()
        cutoff = time.time() - (days * 24 * 60 * 60)

        recent_patterns = []
        seen = set()

        for entry in reversed(self._history):
            if entry.timestamp < cutoff:
                break
            if entry.query_pattern not in seen:
                recent_patterns.append(entry.query_pattern)
                seen.add(entry.query_pattern)
                if len(recent_patterns) >= limit:
                    break

        return recent_patterns

    def clear_history(self) -> None:
        """Clear all search history."""
        self._history.clear()
        if self.history_file.exists():
            self.history_file.unlink()

    def get_stats(self) -> dict[str, int]:
        """Get search history statistics."""
        self.load()
        return {
            "total_searches": len(self._history),
            "unique_patterns": len(set(entry.query_pattern for entry in self._history)),
        }

    def get_detailed_stats(self) -> dict[str, Any]:
        """Get comprehensive search history statistics."""
        self.load()
        if not self._history:
            return {
                "total_searches": 0,
                "unique_patterns": 0,
                "total_elapsed_ms": 0.0,
                "average_elapsed_ms": 0.0,
                "average_success_score": 0.0,
                "total_results": 0,
                "average_results": 0.0,
                "date_range": None,
                "categories": {},
                "storage_bytes": 0,
            }

        entries = list(self._history)
        total = len(entries)
        total_elapsed = sum(e.elapsed_ms for e in entries)
        total_results = sum(e.items_count for e in entries)
        total_score = sum(e.success_score for e in entries)

        category_counts: dict[str, int] = {}
        for e in entries:
            cat = e.category.value if hasattr(e.category, "value") else str(e.category)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        storage_bytes = 0
        if self.history_file.exists():
            storage_bytes = self.history_file.stat().st_size

        return {
            "total_searches": total,
            "unique_patterns": len(set(e.query_pattern for e in entries)),
            "total_elapsed_ms": total_elapsed,
            "average_elapsed_ms": total_elapsed / total,
            "average_success_score": total_score / total,
            "total_results": total_results,
            "average_results": total_results / total,
            "date_range": {
                "earliest": entries[0].timestamp,
                "latest": entries[-1].timestamp,
            },
            "categories": category_counts,
            "storage_bytes": storage_bytes,
        }

    def get_history_by_date_range(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int | None = None,
    ) -> list[SearchHistoryEntry]:
        """Get history entries within a date range."""
        self.load()
        filtered = list(self._history)
        if start_time is not None:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        if end_time is not None:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        filtered = list(reversed(filtered))  # Most recent first
        if limit:
            return filtered[:limit]
        return filtered

    def get_history_by_category(
        self, category: SearchCategory, limit: int | None = None
    ) -> list[SearchHistoryEntry]:
        """Get history entries filtered by category."""
        self.load()
        filtered = [e for e in reversed(self._history) if e.category == category]
        if limit:
            return filtered[:limit]
        return filtered

    def get_history_by_language(
        self, language: str, limit: int | None = None
    ) -> list[SearchHistoryEntry]:
        """Get history entries that involved a specific programming language."""
        self.load()
        lang_lower = language.lower()
        filtered = [
            e for e in reversed(self._history)
            if e.languages and lang_lower in {l.lower() for l in e.languages}
        ]
        if limit:
            return filtered[:limit]
        return filtered

    def search_history(
        self, query: str, limit: int | None = None
    ) -> list[SearchHistoryEntry]:
        """Full-text search across query patterns in history."""
        self.load()
        query_lower = query.lower()
        filtered = [
            e for e in reversed(self._history)
            if query_lower in e.query_pattern.lower()
        ]
        if limit:
            return filtered[:limit]
        return filtered

    def cleanup_old_history(self, days: int) -> int:
        """Remove history entries older than specified days."""
        self.load()
        cutoff = time.time() - (days * 24 * 60 * 60)
        original_len = len(self._history)
        remaining = [e for e in self._history if e.timestamp >= cutoff]
        self._history.clear()
        for e in remaining:
            self._history.append(e)
        removed = original_len - len(self._history)
        if removed > 0:
            self.save_history()
        return removed

    def deduplicate_history(self) -> int:
        """Remove duplicate consecutive entries with the same pattern and close timestamps."""
        self.load()
        if len(self._history) < 2:
            return 0

        entries = list(self._history)
        deduplicated: list[SearchHistoryEntry] = [entries[0]]

        for entry in entries[1:]:
            prev = deduplicated[-1]
            # Consider duplicate if same pattern within 2 seconds
            if (
                entry.query_pattern == prev.query_pattern
                and abs(entry.timestamp - prev.timestamp) < 2.0
            ):
                continue
            deduplicated.append(entry)

        removed = len(entries) - len(deduplicated)
        if removed > 0:
            self._history.clear()
            for e in deduplicated:
                self._history.append(e)
            self.save_history()
        return removed

    # Delegate methods to other managers for backward compatibility
    def add_bookmark(self, name: str, query: Query, result: SearchResult) -> None:
        """Add a search as a bookmark."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            self._bookmark_manager.add_bookmark(name, query, result)

    def get_bookmarks(self) -> dict[str, SearchHistoryEntry]:
        """Get all bookmarks."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.get_bookmarks()  # type: ignore[no-any-return]
        return {}

    def remove_bookmark(self, name: str) -> bool:
        """Remove a bookmark by name."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.remove_bookmark(name)  # type: ignore[no-any-return]
        return False

    def get_current_session(self) -> Any:
        """Get the current search session."""
        self._ensure_managers_loaded()
        if self._session_manager:
            return self._session_manager.get_current_session()
        return None

    def get_sessions(self, limit: int | None = None) -> Any:
        """Get search sessions, most recent first."""
        self._ensure_managers_loaded()
        if self._session_manager:
            return self._session_manager.get_sessions(limit)
        return []

    def get_search_analytics(self, days: int = 30) -> dict[str, Any]:
        """Get comprehensive search analytics."""
        self._ensure_managers_loaded()
        if self._analytics_manager:
            return self._analytics_manager.get_search_analytics(self._history, days)  # type: ignore[no-any-return]
        return {}

    def get_pattern_suggestions(self, partial_pattern: str, limit: int = 5) -> list[str]:
        """Get pattern suggestions based on search history."""
        self._ensure_managers_loaded()
        if self._analytics_manager:
            return self._analytics_manager.get_pattern_suggestions(  # type: ignore[no-any-return]
                self._history, partial_pattern, limit
            )
        return []

    # Bookmark folder delegation
    def create_folder(self, name: str, description: str | None = None) -> bool:
        """Create a new bookmark folder."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.create_folder(name, description)  # type: ignore[no-any-return]
        return False

    def delete_folder(self, name: str) -> bool:
        """Delete a bookmark folder."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.delete_folder(name)  # type: ignore[no-any-return]
        return False

    def add_bookmark_to_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Add a bookmark to a folder."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.add_bookmark_to_folder(bookmark_name, folder_name)  # type: ignore[no-any-return]
        return False

    def remove_bookmark_from_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Remove a bookmark from a folder."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.remove_bookmark_from_folder(bookmark_name, folder_name)  # type: ignore[no-any-return]
        return False

    def get_folders(self) -> dict[str, Any]:
        """Get all bookmark folders."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.get_folders()  # type: ignore[no-any-return]
        return {}

    def get_bookmarks_in_folder(self, folder_name: str) -> list[Any]:
        """Get all bookmarks in a specific folder."""
        self._ensure_managers_loaded()
        if self._bookmark_manager:
            return self._bookmark_manager.get_bookmarks_in_folder(folder_name)  # type: ignore[no-any-return]
        return []

    # Session delegation
    def end_current_session(self) -> None:
        """Manually end the current search session."""
        self._ensure_managers_loaded()
        if self._session_manager:
            self._session_manager.end_current_session()

    # Analytics delegation
    def rate_search(self, pattern: str, rating: int) -> bool:
        """Rate a search result (1-5 stars)."""
        self._ensure_managers_loaded()
        if self._analytics_manager:
            return self._analytics_manager.rate_search(  # type: ignore[no-any-return]
                list(self._history), pattern, rating
            )
        return False

    def add_tags_to_search(self, pattern: str, tags: set[str]) -> bool:
        """Add tags to a search in history."""
        self._ensure_managers_loaded()
        if self._analytics_manager:
            return self._analytics_manager.add_tags_to_search(  # type: ignore[no-any-return]
                list(self._history), pattern, tags
            )
        return False

    def search_history_by_tags(self, tags: set[str]) -> list[SearchHistoryEntry]:
        """Find searches by tags."""
        self._ensure_managers_loaded()
        if self._analytics_manager:
            return self._analytics_manager.search_history_by_tags(  # type: ignore[no-any-return]
                list(self._history), tags
            )
        return []
