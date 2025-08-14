"""
Search history tracking and management for pysearch.

This module provides comprehensive search history functionality including
session tracking, query categorization, performance analytics, and search
pattern analysis. It helps users understand their search behavior and
provides insights for improving search efficiency.

Key Features:
    - Persistent search history storage
    - Query categorization and tagging
    - Performance metrics tracking
    - Search pattern analysis
    - Session management
    - Bookmark and favorite queries
    - Search analytics and insights
    - Query suggestion based on history

Classes:
    SearchCategory: Enumeration of search categories
    SearchHistoryEntry: Individual search record
    SearchSession: Search session information
    SearchHistory: Main history management class

Key Capabilities:
    - Track all search operations with detailed metadata
    - Categorize searches by type (function, class, variable, etc.)
    - Analyze search patterns and suggest improvements
    - Provide search suggestions based on history
    - Export/import history for backup and sharing
    - Generate search analytics and reports

Example:
    Basic history usage:
        >>> from pysearch.history import SearchHistory
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> history = SearchHistory(config)
        >>>
        >>> # Record a search
        >>> history.record_search(query, results)
        >>>
        >>> # Get recent searches
        >>> recent = history.get_recent_searches(limit=10)
        >>> for entry in recent:
        ...     print(f"{entry.query_pattern}: {entry.items_count} results")

    Advanced analytics:
        >>> # Get search analytics
        >>> analytics = history.get_search_analytics()
        >>> print(f"Total searches: {analytics.total_searches}")
        >>> print(f"Average results: {analytics.avg_results_per_search:.1f}")
        >>>
        >>> # Get suggestions based on history
        >>> suggestions = history.get_query_suggestions("def")
        >>> print("Suggested queries:", suggestions)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .config import SearchConfig
from .types import Query, SearchResult


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


@dataclass(slots=True)
class SearchSession:
    """Represents a search session with related queries."""

    session_id: str
    start_time: float
    end_time: float | None = None
    queries: list[str] | None = None
    total_searches: int = 0
    successful_searches: int = 0
    primary_paths: set[str] | None = None
    primary_languages: set[str] | None = None

    def __post_init__(self) -> None:
        if self.queries is None:
            self.queries = []


@dataclass(slots=True)
class BookmarkFolder:
    """Organize bookmarks into folders."""

    name: str
    description: str | None = None
    created_time: float | None = None
    bookmarks: set[str] | None = None

    def __post_init__(self) -> None:
        if self.created_time is None:
            self.created_time = time.time()
        if self.bookmarks is None:
            self.bookmarks = set()


class SearchHistory:
    """Enhanced search history and bookmarks manager with sessions, analytics, and organization."""

    def __init__(self, cfg: SearchConfig, max_entries: int = 1000) -> None:
        self.cfg = cfg
        self.max_entries = max_entries
        self.cache_dir = cfg.resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.cache_dir / "search_history.json"
        self.bookmarks_file = self.cache_dir / "bookmarks.json"
        self.sessions_file = self.cache_dir / "search_sessions.json"
        self.folders_file = self.cache_dir / "bookmark_folders.json"

        self._history: deque[SearchHistoryEntry] = deque(maxlen=max_entries)
        self._bookmarks: dict[str, SearchHistoryEntry] = {}
        self._sessions: dict[str, SearchSession] = {}
        self._folders: dict[str, BookmarkFolder] = {}
        self._current_session: SearchSession | None = None
        self._loaded = False

        # Session management
        self._session_timeout = 30 * 60  # 30 minutes
        self._last_search_time = 0.0

    def load(self) -> None:
        """Load search history, bookmarks, sessions, and folders from disk."""
        if self._loaded:
            return

        # Load history
        if self.history_file.exists():
            try:
                data = json.loads(
                    self.history_file.read_text(encoding="utf-8"))
                entries = data.get("entries", [])
                # Keep only recent entries
                for entry_data in entries[-self.max_entries:]:
                    # Handle legacy entries without new fields
                    if "category" not in entry_data:
                        entry_data["category"] = SearchCategory.GENERAL.value
                    if "languages" in entry_data and entry_data["languages"]:
                        entry_data["languages"] = set(entry_data["languages"])
                    if "tags" in entry_data and entry_data["tags"]:
                        entry_data["tags"] = set(entry_data["tags"])

                    entry = SearchHistoryEntry(**entry_data)
                    self._history.append(entry)
            except Exception:
                pass

        # Load bookmarks
        if self.bookmarks_file.exists():
            try:
                data = json.loads(
                    self.bookmarks_file.read_text(encoding="utf-8"))
                bookmarks = data.get("bookmarks", {})
                for name, entry_data in bookmarks.items():
                    # Handle legacy entries
                    if "category" not in entry_data:
                        entry_data["category"] = SearchCategory.GENERAL.value
                    if "languages" in entry_data and entry_data["languages"]:
                        entry_data["languages"] = set(entry_data["languages"])
                    if "tags" in entry_data and entry_data["tags"]:
                        entry_data["tags"] = set(entry_data["tags"])

                    entry = SearchHistoryEntry(**entry_data)
                    self._bookmarks[name] = entry
            except Exception:
                pass

        # Load sessions
        if self.sessions_file.exists():
            try:
                data = json.loads(
                    self.sessions_file.read_text(encoding="utf-8"))
                sessions = data.get("sessions", {})
                for session_id, session_data in sessions.items():
                    if "primary_paths" in session_data and session_data["primary_paths"]:
                        session_data["primary_paths"] = set(
                            session_data["primary_paths"])
                    if "primary_languages" in session_data and session_data["primary_languages"]:
                        session_data["primary_languages"] = set(
                            session_data["primary_languages"])

                    session = SearchSession(**session_data)
                    self._sessions[session_id] = session
            except Exception:
                pass

        # Load bookmark folders
        if self.folders_file.exists():
            try:
                data = json.loads(
                    self.folders_file.read_text(encoding="utf-8"))
                folders = data.get("folders", {})
                for folder_name, folder_data in folders.items():
                    if "bookmarks" in folder_data and folder_data["bookmarks"]:
                        folder_data["bookmarks"] = set(
                            folder_data["bookmarks"])

                    folder = BookmarkFolder(**folder_data)
                    self._folders[folder_name] = folder
            except Exception:
                pass

        self._loaded = True

    def add_search(self, query: Query, result: SearchResult) -> None:
        """Add a search to history with enhanced tracking."""
        self.load()
        current_time = time.time()

        # Manage search sessions
        session = self._get_or_create_session(current_time)

        # Categorize the search
        category = self._categorize_search(query)

        # Calculate success score based on results
        success_score = self._calculate_success_score(result)

        # Extract languages from results
        languages = self._extract_languages_from_results(result)

        # Extract paths
        paths = [str(item.file.parent)
                 for item in result.items[:10]]  # Top 10 paths

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
            session_id=session.session_id,
            category=category,
            languages=languages,
            paths=paths,
            success_score=success_score,
        )

        self._history.append(entry)
        self._update_session(session, query, result)
        self._last_search_time = current_time

        self.save_history()
        self.save_sessions()

    def _get_or_create_session(self, current_time: float) -> SearchSession:
        """Get current session or create a new one."""
        # Check if we need a new session (timeout or no current session)
        if (
            self._current_session is None
            or current_time - self._last_search_time > self._session_timeout
        ):

            # End current session if exists
            if self._current_session:
                self._current_session.end_time = self._last_search_time

            # Create new session
            session_id = hashlib.md5(
                f"{current_time}".encode()).hexdigest()[:8]
            self._current_session = SearchSession(
                session_id=session_id, start_time=current_time)
            self._sessions[session_id] = self._current_session

        return self._current_session

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
        languages = set()

        for item in result.items[:20]:  # Check top 20 results
            ext = item.file.suffix.lower()
            if ext == ".py":
                languages.add("python")
            elif ext in [".js", ".jsx"]:
                languages.add("javascript")
            elif ext in [".ts", ".tsx"]:
                languages.add("typescript")
            elif ext == ".java":
                languages.add("java")
            elif ext in [".c", ".h"]:
                languages.add("c")
            elif ext in [".cpp", ".cc", ".cxx", ".hpp"]:
                languages.add("cpp")
            elif ext == ".go":
                languages.add("go")
            elif ext == ".rs":
                languages.add("rust")
            elif ext == ".rb":
                languages.add("ruby")
            elif ext == ".php":
                languages.add("php")

        return languages

    def _update_session(self, session: SearchSession, query: Query, result: SearchResult) -> None:
        """Update session statistics."""
        session.total_searches += 1
        if result.stats.items > 0:
            session.successful_searches += 1

        if session.queries:
            session.queries.append(query.pattern)
        else:
            session.queries = [query.pattern]

        # Update primary paths and languages
        if result.items:
            paths = {str(item.file.parent) for item in result.items[:10]}
            if session.primary_paths:
                session.primary_paths.update(paths)
            else:
                session.primary_paths = paths

            languages = self._extract_languages_from_results(result)
            if session.primary_languages:
                session.primary_languages.update(languages)
            else:
                session.primary_languages = languages

    def save_history(self) -> None:
        """Save search history to disk."""
        try:
            data = {
                "version": 1,
                "last_updated": time.time(),
                "entries": [asdict(entry) for entry in self._history],
            }
            tmp_file = self.history_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(
                data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.history_file)
        except Exception:
            pass

    def save_bookmarks(self) -> None:
        """Save bookmarks to disk."""
        try:
            data = {
                "version": 1,
                "last_updated": time.time(),
                "bookmarks": {name: asdict(entry) for name, entry in self._bookmarks.items()},
            }
            tmp_file = self.bookmarks_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(
                data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.bookmarks_file)
        except Exception:
            pass

    def save_sessions(self) -> None:
        """Save search sessions to disk."""
        try:
            # Convert sets to lists for JSON serialization
            sessions_data = {}
            for session_id, session in self._sessions.items():
                session_dict = asdict(session)
                if session_dict.get("primary_paths"):
                    session_dict["primary_paths"] = list(
                        session_dict["primary_paths"])
                if session_dict.get("primary_languages"):
                    session_dict["primary_languages"] = list(
                        session_dict["primary_languages"])
                sessions_data[session_id] = session_dict

            data = {"version": 1, "last_updated": time.time(),
                    "sessions": sessions_data}
            tmp_file = self.sessions_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(
                data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.sessions_file)
        except Exception:
            pass

    def save_folders(self) -> None:
        """Save bookmark folders to disk."""
        try:
            # Convert sets to lists for JSON serialization
            folders_data = {}
            for folder_name, folder in self._folders.items():
                folder_dict = asdict(folder)
                if folder_dict.get("bookmarks"):
                    folder_dict["bookmarks"] = list(folder_dict["bookmarks"])
                folders_data[folder_name] = folder_dict

            data = {"version": 1, "last_updated": time.time(),
                    "folders": folders_data}
            tmp_file = self.folders_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(
                data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.folders_file)
        except Exception:
            pass

    def add_bookmark(self, name: str, query: Query, result: SearchResult) -> None:
        """Add a search as a bookmark."""
        self.load()

        entry = SearchHistoryEntry(
            timestamp=time.time(),
            query_pattern=query.pattern,
            use_regex=query.use_regex,
            use_ast=query.use_ast,
            context=query.context,
            files_matched=result.stats.files_matched,
            items_count=result.stats.items,
            elapsed_ms=result.stats.elapsed_ms,
            filters=str(query.filters) if query.filters else None,
        )

        self._bookmarks[name] = entry
        self.save_bookmarks()

    def get_history(self, limit: int | None = None) -> list[SearchHistoryEntry]:
        """Get search history, most recent first."""
        self.load()
        history_list = list(reversed(self._history))
        if limit:
            return history_list[:limit]
        return history_list

    def get_bookmarks(self) -> dict[str, SearchHistoryEntry]:
        """Get all bookmarks."""
        self.load()
        return dict(self._bookmarks)

    def remove_bookmark(self, name: str) -> bool:
        """Remove a bookmark by name."""
        self.load()
        if name in self._bookmarks:
            del self._bookmarks[name]
            self.save_bookmarks()
            return True
        return False

    def get_frequent_patterns(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most frequently used search patterns."""
        self.load()
        pattern_counts: dict[str, int] = {}

        for entry in self._history:
            pattern = entry.query_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        sorted_patterns = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True)
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

    # Enhanced session management methods
    def get_current_session(self) -> SearchSession | None:
        """Get the current search session."""
        self.load()
        return self._current_session

    def get_sessions(self, limit: int | None = None) -> list[SearchSession]:
        """Get search sessions, most recent first."""
        self.load()
        sessions = sorted(self._sessions.values(),
                          key=lambda s: s.start_time, reverse=True)
        return sessions[:limit] if limit else sessions

    def get_session_by_id(self, session_id: str) -> SearchSession | None:
        """Get a specific session by ID."""
        self.load()
        return self._sessions.get(session_id)

    def end_current_session(self) -> None:
        """Manually end the current session."""
        if self._current_session:
            self._current_session.end_time = time.time()
            self.save_sessions()
            self._current_session = None

    # Enhanced bookmark organization methods
    def create_folder(self, name: str, description: str | None = None) -> bool:
        """Create a new bookmark folder."""
        self.load()
        if name in self._folders:
            return False

        self._folders[name] = BookmarkFolder(
            name=name, description=description)
        self.save_folders()
        return True

    def delete_folder(self, name: str) -> bool:
        """Delete a bookmark folder."""
        self.load()
        if name not in self._folders:
            return False

        del self._folders[name]
        self.save_folders()
        return True

    def add_bookmark_to_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Add a bookmark to a folder."""
        self.load()
        if bookmark_name not in self._bookmarks or folder_name not in self._folders:
            return False

        folder = self._folders[folder_name]
        if folder.bookmarks is None:
            folder.bookmarks = set()
        folder.bookmarks.add(bookmark_name)
        self.save_folders()
        return True

    def remove_bookmark_from_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Remove a bookmark from a folder."""
        self.load()
        if folder_name not in self._folders:
            return False

        folder = self._folders[folder_name]
        if folder.bookmarks and bookmark_name in folder.bookmarks:
            folder.bookmarks.remove(bookmark_name)
            self.save_folders()
            return True
        return False

    def get_folders(self) -> dict[str, BookmarkFolder]:
        """Get all bookmark folders."""
        self.load()
        return self._folders.copy()

    def get_bookmarks_in_folder(self, folder_name: str) -> list[SearchHistoryEntry]:
        """Get all bookmarks in a specific folder."""
        self.load()
        if folder_name not in self._folders:
            return []

        folder = self._folders[folder_name]
        if not folder.bookmarks:
            return []

        return [self._bookmarks[name] for name in folder.bookmarks if name in self._bookmarks]

    # Enhanced analytics methods
    def get_search_analytics(self, days: int = 30) -> dict[str, Any]:
        """Get comprehensive search analytics."""
        self.load()
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        recent_entries = [
            entry for entry in self._history if entry.timestamp >= cutoff_time]

        if not recent_entries:
            return {
                "total_searches": 0,
                "successful_searches": 0,
                "average_success_score": 0.0,
                "most_common_categories": [],
                "most_used_languages": [],
                "average_search_time": 0.0,
                "search_frequency": {},
                "session_count": 0,
            }

        # Calculate statistics
        total_searches = len(recent_entries)
        successful_searches = sum(
            1 for entry in recent_entries if entry.items_count > 0)
        average_success_score = (
            sum(entry.success_score for entry in recent_entries) / total_searches
        )
        average_search_time = sum(
            entry.elapsed_ms for entry in recent_entries) / total_searches

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
        most_used_languages = sorted(
            language_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Search frequency by day
        search_frequency: dict[str, int] = defaultdict(int)
        for entry in recent_entries:
            day = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            search_frequency[day] += 1

        # Session count
        recent_sessions = [
            s for s in self._sessions.values() if s.start_time >= cutoff_time]

        return {
            "total_searches": total_searches,
            "successful_searches": successful_searches,
            "success_rate": successful_searches / total_searches if total_searches > 0 else 0,
            "average_success_score": average_success_score,
            "most_common_categories": most_common_categories,
            "most_used_languages": most_used_languages,
            "average_search_time": average_search_time,
            "search_frequency": dict(search_frequency),
            "session_count": len(recent_sessions),
        }

    def get_pattern_suggestions(self, partial_pattern: str, limit: int = 5) -> list[str]:
        """Get pattern suggestions based on search history."""
        self.load()
        partial_lower = partial_pattern.lower()

        suggestions = []
        seen = set()

        # Look for patterns that start with or contain the partial pattern
        for entry in reversed(self._history):  # Most recent first
            pattern = entry.query_pattern
            if (
                pattern.lower().startswith(partial_lower) or partial_lower in pattern.lower()
            ) and pattern not in seen:
                suggestions.append(pattern)
                seen.add(pattern)

                if len(suggestions) >= limit:
                    break

        return suggestions

    def rate_search(self, pattern: str, rating: int) -> bool:
        """Rate a search result (1-5 stars)."""
        if not 1 <= rating <= 5:
            return False

        self.load()

        # Find the most recent search with this pattern
        for entry in reversed(self._history):
            if entry.query_pattern == pattern:
                entry.user_rating = rating
                self.save_history()
                return True

        return False

    def add_tags_to_search(self, pattern: str, tags: set[str]) -> bool:
        """Add tags to a search in history."""
        self.load()

        # Find the most recent search with this pattern
        for entry in reversed(self._history):
            if entry.query_pattern == pattern:
                if entry.tags:
                    entry.tags.update(tags)
                else:
                    entry.tags = tags.copy()
                self.save_history()
                return True

        return False

    def search_history_by_tags(self, tags: set[str]) -> list[SearchHistoryEntry]:
        """Find searches by tags."""
        self.load()

        matching_entries = []
        for entry in self._history:
            if entry.tags and tags.intersection(entry.tags):
                matching_entries.append(entry)

        return sorted(matching_entries, key=lambda e: e.timestamp, reverse=True)

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
            "total_bookmarks": len(self._bookmarks),
            "unique_patterns": len(set(entry.query_pattern for entry in self._history)),
        }
