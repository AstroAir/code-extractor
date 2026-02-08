"""
Search session tracking and management functionality.

This module provides comprehensive search session management capabilities,
including session creation, tracking, and analytics for understanding
search patterns and user behavior.

Classes:
    SearchSession: Represents a search session with related queries
    SessionManager: Main session management class

Key Features:
    - Track search sessions with automatic timeout
    - Group related searches into sessions
    - Session-based analytics and insights
    - Persistent storage of session data

Example:
    Session management:
        >>> from pysearch.core.history.history_sessions import SessionManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = SessionManager(config)
        >>>
        >>> # Get current session
        >>> session = manager.get_current_session()
        >>> print(f"Session {session.session_id}: {session.total_searches} searches")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..config import SearchConfig
from ..types import Query, SearchResult


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


class SessionManager:
    """Search session tracking and management."""

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg
        self.cache_dir = cfg.resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_file = self.cache_dir / "search_sessions.json"

        self._sessions: dict[str, SearchSession] = {}
        self._current_session: SearchSession | None = None
        self._loaded = False

        # Session management
        self._session_timeout = 30 * 60  # 30 minutes
        self._last_search_time = 0.0

    def load(self) -> None:
        """Load search sessions from disk."""
        if self._loaded:
            return

        if self.sessions_file.exists():
            try:
                data = json.loads(self.sessions_file.read_text(encoding="utf-8"))
                sessions = data.get("sessions", {})
                for session_id, session_data in sessions.items():
                    if "primary_paths" in session_data and session_data["primary_paths"]:
                        session_data["primary_paths"] = set(session_data["primary_paths"])
                    if "primary_languages" in session_data and session_data["primary_languages"]:
                        session_data["primary_languages"] = set(session_data["primary_languages"])

                    session = SearchSession(**session_data)
                    self._sessions[session_id] = session
            except Exception:
                pass

        self._loaded = True

    def save_sessions(self) -> None:
        """Save search sessions to disk."""
        try:
            # Convert sets to lists for JSON serialization
            sessions_data = {}
            for session_id, session in self._sessions.items():
                session_dict = asdict(session)
                if session_dict.get("primary_paths"):
                    session_dict["primary_paths"] = list(session_dict["primary_paths"])
                if session_dict.get("primary_languages"):
                    session_dict["primary_languages"] = list(session_dict["primary_languages"])
                sessions_data[session_id] = session_dict

            data = {"version": 1, "last_updated": time.time(), "sessions": sessions_data}
            tmp_file = self.sessions_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self.sessions_file)
        except Exception:
            pass

    def get_or_create_session(self, current_time: float) -> SearchSession:
        """Get current session or create a new one."""
        self.load()

        # Check if we need a new session (timeout or no current session)
        if (
            self._current_session is None
            or current_time - self._last_search_time > self._session_timeout
        ):

            # End current session if exists
            if self._current_session:
                self._current_session.end_time = self._last_search_time

            # Create new session
            session_id = hashlib.md5(f"{current_time}".encode()).hexdigest()[:8]
            self._current_session = SearchSession(session_id=session_id, start_time=current_time)
            self._sessions[session_id] = self._current_session

        return self._current_session

    def update_session(self, session: SearchSession, query: Query, result: SearchResult) -> None:
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

        # Update last search time
        self._last_search_time = time.time()

    def _extract_languages_from_results(self, result: SearchResult) -> set[str]:
        """Extract programming languages from search results."""
        from . import extract_languages_from_results

        return extract_languages_from_results(result)

    def get_current_session(self) -> SearchSession | None:
        """Get the current search session."""
        self.load()
        return self._current_session

    def get_sessions(self, limit: int | None = None) -> list[SearchSession]:
        """Get search sessions, most recent first."""
        self.load()
        sessions = sorted(self._sessions.values(), key=lambda s: s.start_time, reverse=True)
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

    def get_session_analytics(self, days: int = 30) -> dict[str, Any]:
        """Get session-based analytics."""
        self.load()
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        recent_sessions = [s for s in self._sessions.values() if s.start_time >= cutoff_time]

        if not recent_sessions:
            return {
                "total_sessions": 0,
                "average_searches_per_session": 0.0,
                "average_session_duration": 0.0,
                "most_active_session": None,
            }

        total_searches = sum(s.total_searches for s in recent_sessions)
        total_duration = sum((s.end_time or time.time()) - s.start_time for s in recent_sessions)

        # Find most active session
        most_active = max(recent_sessions, key=lambda s: s.total_searches)

        return {
            "total_sessions": len(recent_sessions),
            "average_searches_per_session": total_searches / len(recent_sessions),
            "average_session_duration": total_duration / len(recent_sessions),
            "most_active_session": {
                "session_id": most_active.session_id,
                "searches": most_active.total_searches,
                "duration": (most_active.end_time or time.time()) - most_active.start_time,
            },
        }

    def cleanup_old_sessions(self, days: int = 90) -> int:
        """Remove sessions older than specified days."""
        self.load()
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        old_sessions = [
            session_id
            for session_id, session in self._sessions.items()
            if session.start_time < cutoff_time
        ]

        for session_id in old_sessions:
            del self._sessions[session_id]

        if old_sessions:
            self.save_sessions()

        return len(old_sessions)

    def compare_sessions(
        self, session_id_1: str, session_id_2: str
    ) -> dict[str, Any]:
        """
        Compare two sessions and return their differences.

        Args:
            session_id_1: First session ID
            session_id_2: Second session ID

        Returns:
            Dictionary with comparison data
        """
        self.load()
        s1 = self._sessions.get(session_id_1)
        s2 = self._sessions.get(session_id_2)

        if not s1 or not s2:
            missing = []
            if not s1:
                missing.append(session_id_1)
            if not s2:
                missing.append(session_id_2)
            return {"error": f"Session(s) not found: {', '.join(missing)}"}

        s1_queries = set(s1.queries or [])
        s2_queries = set(s2.queries or [])
        s1_langs = s1.primary_languages or set()
        s2_langs = s2.primary_languages or set()
        s1_paths = s1.primary_paths or set()
        s2_paths = s2.primary_paths or set()

        s1_duration = (s1.end_time or time.time()) - s1.start_time
        s2_duration = (s2.end_time or time.time()) - s2.start_time

        s1_success_rate = (
            s1.successful_searches / s1.total_searches
            if s1.total_searches > 0
            else 0.0
        )
        s2_success_rate = (
            s2.successful_searches / s2.total_searches
            if s2.total_searches > 0
            else 0.0
        )

        return {
            "session_1": {
                "session_id": session_id_1,
                "total_searches": s1.total_searches,
                "successful_searches": s1.successful_searches,
                "success_rate": s1_success_rate,
                "duration_seconds": s1_duration,
                "query_count": len(s1_queries),
            },
            "session_2": {
                "session_id": session_id_2,
                "total_searches": s2.total_searches,
                "successful_searches": s2.successful_searches,
                "success_rate": s2_success_rate,
                "duration_seconds": s2_duration,
                "query_count": len(s2_queries),
            },
            "common_queries": sorted(s1_queries & s2_queries),
            "only_in_session_1": sorted(s1_queries - s2_queries),
            "only_in_session_2": sorted(s2_queries - s1_queries),
            "common_languages": sorted(s1_langs & s2_langs),
            "common_paths": sorted(s1_paths & s2_paths),
            "search_count_diff": s2.total_searches - s1.total_searches,
            "success_rate_diff": s2_success_rate - s1_success_rate,
            "duration_diff_seconds": s2_duration - s1_duration,
        }

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """
        Get a detailed summary of a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with detailed session summary
        """
        self.load()
        session = self._sessions.get(session_id)
        if not session:
            return {"error": f"Session not found: {session_id}"}

        duration = (session.end_time or time.time()) - session.start_time
        success_rate = (
            session.successful_searches / session.total_searches
            if session.total_searches > 0
            else 0.0
        )

        # Analyze query patterns
        queries = session.queries or []
        unique_queries = set(queries)
        repeated_queries = [q for q in unique_queries if queries.count(q) > 1]

        from datetime import datetime

        return {
            "session_id": session.session_id,
            "start_time": session.start_time,
            "start_time_iso": datetime.fromtimestamp(session.start_time).isoformat(),
            "end_time": session.end_time,
            "end_time_iso": (
                datetime.fromtimestamp(session.end_time).isoformat()
                if session.end_time
                else None
            ),
            "is_active": session.end_time is None,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "total_searches": session.total_searches,
            "successful_searches": session.successful_searches,
            "failed_searches": session.total_searches - session.successful_searches,
            "success_rate": success_rate,
            "total_queries": len(queries),
            "unique_queries": len(unique_queries),
            "repeated_queries": repeated_queries,
            "primary_languages": sorted(session.primary_languages) if session.primary_languages else [],
            "primary_paths": sorted(session.primary_paths) if session.primary_paths else [],
            "searches_per_minute": (
                session.total_searches / (duration / 60) if duration > 0 else 0.0
            ),
        }
