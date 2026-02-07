#!/usr/bin/env python3
"""
Enhanced Session Management for PySearch MCP Server

This module provides sophisticated session management capabilities including:
- Context-aware search sessions
- User preferences and personalization
- Cross-search correlation and learning
- Session analytics and insights
- Persistent session storage
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Import removed due to circular dependency
# from ..servers.mcp_server import SearchSession


# Simple session data structure
@dataclass
class SearchSession:
    """Simple search session data structure."""

    session_id: str
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)


class SessionPriority(Enum):
    """Session priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SearchIntent(Enum):
    """Types of search intent for better context understanding."""

    EXPLORATION = "exploration"  # User exploring codebase
    DEBUGGING = "debugging"  # User debugging an issue
    REFACTORING = "refactoring"  # User refactoring code
    LEARNING = "learning"  # User learning about code
    MAINTENANCE = "maintenance"  # User maintaining code
    UNKNOWN = "unknown"  # Intent unclear


class UserProfile:
    """User profile for personalized search experience."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.created_at = datetime.now()
        self.preferences: dict[str, Any] = {}
        self.search_patterns: dict[str, int] = {}  # Pattern frequency
        self.language_preferences: dict[str, float] = {}  # Language usage weights
        self.common_paths: dict[str, int] = {}  # Frequently searched paths
        self.session_count: int = 0
        self.total_search_time: float = 0.0
        self.successful_searches: int = 0
        self.failed_searches: int = 0
        self.preferred_languages: list[str] = []

    def update_preferences(self, **preferences: Any) -> None:
        """Update user preferences."""
        self.preferences.update(preferences)

    def record_search_pattern(self, pattern: str) -> None:
        """Record a search pattern for learning user behavior."""
        self.search_patterns[pattern] = self.search_patterns.get(pattern, 0) + 1

    def record_language_usage(self, language: str, weight: float = 1.0) -> None:
        """Record language usage for personalization."""
        current = self.language_preferences.get(language, 0.0)
        self.language_preferences[language] = current + weight

    def record_path_usage(self, path: str) -> None:
        """Record path usage for intelligent defaults."""
        self.common_paths[path] = self.common_paths.get(path, 0) + 1

    def get_preferred_languages(self, top_k: int = 5) -> list[str]:
        """Get user's preferred languages by usage."""
        sorted_langs = sorted(self.language_preferences.items(), key=lambda x: x[1], reverse=True)
        return [lang for lang, _ in sorted_langs[:top_k]]

    def get_common_patterns(self, top_k: int = 10) -> list[str]:
        """Get user's most common search patterns."""
        sorted_patterns = sorted(self.search_patterns.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, _ in sorted_patterns[:top_k]]

    def get_suggested_paths(self, top_k: int = 5) -> list[str]:
        """Get suggested paths based on user history."""
        sorted_paths = sorted(self.common_paths.items(), key=lambda x: x[1], reverse=True)
        return [path for path, _ in sorted_paths[:top_k]]


@dataclass
class EnhancedSearchSession:
    """Enhanced search session with context awareness."""

    session_id: str
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    priority: SessionPriority = field(default=SessionPriority.NORMAL)
    intent: SearchIntent = field(default=SearchIntent.UNKNOWN)

    # Search context
    queries: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    cached_results: dict[str, Any] = field(default_factory=dict)

    # Context awareness
    current_files: set[str] = field(default_factory=set)  # Files being worked on
    search_focus: str | None = field(default=None)  # Current search focus area
    recent_patterns: list[str] = field(default_factory=list)  # Recent search patterns
    correlation_context: dict[str, Any] = field(default_factory=dict)  # Cross-search context

    # Analytics
    total_searches: int = field(default=0)
    successful_searches: int = field(default=0)
    avg_search_time: float = field(default=0.0)
    patterns_discovered: set[str] = field(default_factory=set)

    def update_access_time(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.now()

    def add_query(self, query_info: dict[str, Any]) -> None:
        """Add a query to the session with context tracking."""
        self.queries.append(
            {
                **query_info,
                "timestamp": datetime.now().isoformat(),
                "session_context": self.get_current_context(),
            }
        )

        # Update analytics
        self.total_searches += 1
        pattern = query_info.get("pattern", "")
        if pattern:
            self.recent_patterns.append(pattern)
            self.patterns_discovered.add(pattern)

            # Keep recent patterns manageable
            if len(self.recent_patterns) > 20:
                self.recent_patterns = self.recent_patterns[-10:]

    def add_successful_search(self, execution_time: float, result_count: int) -> None:
        """Record a successful search."""
        self.successful_searches += 1

        # Update average search time
        total_time = self.avg_search_time * (self.successful_searches - 1) + execution_time
        self.avg_search_time = total_time / self.successful_searches

        # Update context based on results
        if result_count > 0:
            self.context["last_successful_search"] = datetime.now().isoformat()
            self.context["recent_result_count"] = result_count

    def set_search_focus(self, focus: str) -> None:
        """Set the current search focus."""
        self.search_focus = focus
        self.context["search_focus"] = focus
        self.context["focus_changed_at"] = datetime.now().isoformat()

    def add_file_context(self, file_path: str) -> None:
        """Add a file to the current working context."""
        self.current_files.add(file_path)

        # Keep context manageable
        if len(self.current_files) > 50:
            # Remove oldest files (simplified - could be more sophisticated)
            files_list = list(self.current_files)
            self.current_files = set(files_list[-25:])

    def get_current_context(self) -> dict[str, Any]:
        """Get current session context snapshot."""
        return {
            "session_id": self.session_id,
            "intent": self.intent.value,
            "search_focus": self.search_focus,
            "current_files_count": len(self.current_files),
            "recent_patterns_count": len(self.recent_patterns),
            "total_searches": self.total_searches,
            "success_rate": self.successful_searches / max(self.total_searches, 1),
            "patterns_discovered_count": len(self.patterns_discovered),
        }

    def infer_intent(self) -> SearchIntent:
        """Infer search intent from query patterns."""
        if not self.queries:
            return SearchIntent.UNKNOWN

        recent_queries = self.queries[-5:]  # Last 5 queries
        patterns = [q.get("pattern", "").lower() for q in recent_queries]

        # Simple heuristics for intent detection
        debug_keywords = ["error", "exception", "bug", "fail", "crash", "debug"]
        refactor_keywords = ["class", "function", "method", "rename", "move"]
        learning_keywords = ["example", "how", "what", "why", "learn"]

        debug_score = sum(1 for p in patterns for k in debug_keywords if k in p)
        refactor_score = sum(1 for p in patterns for k in refactor_keywords if k in p)
        learning_score = sum(1 for p in patterns for k in learning_keywords if k in p)

        if debug_score > max(refactor_score, learning_score):
            return SearchIntent.DEBUGGING
        elif refactor_score > max(debug_score, learning_score):
            return SearchIntent.REFACTORING
        elif learning_score > max(debug_score, refactor_score):
            return SearchIntent.LEARNING

        # Check for exploration patterns
        if len(set(patterns)) > len(patterns) * 0.7:  # High pattern diversity
            return SearchIntent.EXPLORATION

        return SearchIntent.UNKNOWN

    def get_contextual_suggestions(self) -> dict[str, Any]:
        """Get contextual suggestions based on session state."""
        suggestions: dict[str, Any] = {
            "suggested_paths": [],
            "suggested_patterns": [],
            "related_files": [],
            "next_steps": [],
        }

        # Suggest paths based on current files
        if self.current_files:
            # Get common parent directories
            paths = [str(Path(f).parent) for f in self.current_files]
            path_counts: dict[str, int] = {}
            for path in paths:
                path_counts[path] = path_counts.get(path, 0) + 1

            suggestions["suggested_paths"] = sorted(
                path_counts.keys(), key=lambda x: path_counts[x], reverse=True
            )[:5]

        # Suggest patterns based on recent usage
        if self.recent_patterns:
            # Find similar patterns
            pattern_counts: dict[str, int] = {}
            for pattern in self.recent_patterns:
                words = pattern.split()
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        pattern_counts[word] = pattern_counts.get(word, 0) + 1

            suggestions["suggested_patterns"] = sorted(
                pattern_counts.keys(), key=lambda x: pattern_counts[x], reverse=True
            )[:5]

        # Intent-based suggestions
        if self.intent == SearchIntent.DEBUGGING:
            suggestions["next_steps"] = [
                "Search for error handling patterns",
                "Look for test files related to current code",
                "Check for logging statements",
            ]
        elif self.intent == SearchIntent.REFACTORING:
            suggestions["next_steps"] = [
                "Find all usages of target symbols",
                "Identify dependencies and imports",
                "Look for similar patterns to refactor",
            ]
        elif self.intent == SearchIntent.LEARNING:
            suggestions["next_steps"] = [
                "Find example usages",
                "Look for documentation and comments",
                "Search for test cases",
            ]

        return suggestions


class EnhancedSessionManager:
    """Enhanced session manager with context awareness and learning."""

    def __init__(self, persist_sessions: bool = True, session_file: str | None = None) -> None:
        self.sessions: dict[str, EnhancedSearchSession] = {}
        self.user_profiles: dict[str, UserProfile] = {}
        self.persist_sessions = persist_sessions
        self.session_file: str = session_file or "session_data.json"
        self.correlation_graph: dict[str, set[str]] = {}  # Pattern correlations

        # Load persisted sessions
        if self.persist_sessions:
            self._load_sessions()

    def create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        priority: SessionPriority = SessionPriority.NORMAL,
    ) -> EnhancedSearchSession:
        """Create a new enhanced search session."""
        import hashlib

        if session_id is None:
            timestamp = str(time.time())
            session_id = hashlib.md5(timestamp.encode()).hexdigest()[:12]

        session = EnhancedSearchSession(session_id=session_id, user_id=user_id, priority=priority)

        self.sessions[session_id] = session

        # Update user profile
        if user_id:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id)
            self.user_profiles[user_id].session_count += 1

        # Persist if enabled
        if self.persist_sessions:
            self._save_sessions()

        return session

    def get_session(self, session_id: str) -> EnhancedSearchSession | None:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.update_access_time()
        return session

    def get_or_create_session(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> EnhancedSearchSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_access_time()
            return session

        return self.create_session(session_id, user_id)

    def record_search(
        self,
        session_id: str,
        query_info: dict[str, Any],
        execution_time: float,
        result_count: int,
        success: bool = True,
    ) -> None:
        """Record a search in the session with learning."""
        session = self.get_session(session_id)
        if not session:
            return

        # Record in session
        session.add_query(query_info)

        if success and result_count > 0:
            session.add_successful_search(execution_time, result_count)

        # Update user profile
        if session.user_id and session.user_id in self.user_profiles:
            profile = self.user_profiles[session.user_id]
            pattern = query_info.get("pattern", "")
            if pattern:
                profile.record_search_pattern(pattern)

            # Record timing
            profile.total_search_time += execution_time
            if success:
                profile.successful_searches += 1
            else:
                profile.failed_searches += 1

        # Update pattern correlations
        pattern = query_info.get("pattern", "")
        if pattern and session.recent_patterns:
            self._update_pattern_correlations(pattern, session.recent_patterns)

        # Infer intent
        new_intent = session.infer_intent()
        if new_intent != session.intent:
            session.intent = new_intent

        # Persist changes
        if self.persist_sessions:
            self._save_sessions()

    def get_contextual_recommendations(self, session_id: str) -> dict[str, Any]:
        """Get contextual recommendations for a session."""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        recommendations = session.get_contextual_suggestions()

        # Add user profile based suggestions
        if session.user_id and session.user_id in self.user_profiles:
            profile = self.user_profiles[session.user_id]
            recommendations["user_preferences"] = {
                "common_patterns": profile.get_common_patterns(5),
                "preferred_languages": profile.get_preferred_languages(3),
                "suggested_paths": profile.get_suggested_paths(3),
            }

        # Add correlation-based suggestions
        if session.recent_patterns:
            latest_pattern = session.recent_patterns[-1]
            correlated = self._get_correlated_patterns(latest_pattern)
            recommendations["related_patterns"] = correlated[:5]

        return recommendations

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive sessions."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)

        old_sessions = []
        for session_id, session in self.sessions.items():
            if session.last_accessed < cutoff_time and session.priority != SessionPriority.CRITICAL:
                old_sessions.append(session_id)

        for session_id in old_sessions:
            del self.sessions[session_id]

        if self.persist_sessions and old_sessions:
            self._save_sessions()

        return len(old_sessions)

    def get_session_analytics(self) -> dict[str, Any]:
        """Get comprehensive session analytics."""
        total_sessions = len(self.sessions)
        active_sessions = len(
            [s for s in self.sessions.values() if (datetime.now() - s.last_accessed).seconds < 3600]
        )

        # Intent distribution
        intent_counts: dict[str, int] = {}
        for session in self.sessions.values():
            intent = session.intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # User analytics
        user_analytics = {}
        for user_id, profile in self.user_profiles.items():
            user_analytics[user_id] = {
                "sessions": profile.session_count,
                "total_searches": sum(profile.search_patterns.values()),
                "success_rate": profile.successful_searches
                / max(profile.successful_searches + profile.failed_searches, 1),
                "preferred_languages": profile.get_preferred_languages(3),
            }

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "intent_distribution": intent_counts,
            "total_users": len(self.user_profiles),
            "pattern_correlations": len(self.correlation_graph),
            "user_analytics": user_analytics,
        }

    def _update_pattern_correlations(self, pattern: str, recent_patterns: list[str]) -> None:
        """Update pattern correlation graph."""
        if pattern not in self.correlation_graph:
            self.correlation_graph[pattern] = set()

        # Add correlations with recent patterns
        for recent_pattern in recent_patterns[-5:]:  # Last 5 patterns
            if recent_pattern != pattern:
                self.correlation_graph[pattern].add(recent_pattern)

                # Bidirectional correlation
                if recent_pattern not in self.correlation_graph:
                    self.correlation_graph[recent_pattern] = set()
                self.correlation_graph[recent_pattern].add(pattern)

    def _get_correlated_patterns(self, pattern: str) -> list[str]:
        """Get patterns correlated with the given pattern."""
        if pattern not in self.correlation_graph:
            return []

        correlations = self.correlation_graph[pattern]

        # Sort by correlation strength (simplified - could be more sophisticated)
        return list(correlations)

    def _save_sessions(self) -> None:
        """Save sessions to persistent storage."""
        try:
            data: dict[str, Any] = {"sessions": {}, "user_profiles": {}, "correlation_graph": {}}

            # Convert sessions to serializable format
            for session_id, session in self.sessions.items():
                session_data = asdict(session)
                # Convert sets to lists for JSON serialization
                session_data["current_files"] = list(session.current_files)
                session_data["patterns_discovered"] = list(session.patterns_discovered)
                data["sessions"][session_id] = session_data

            # Convert user profiles
            for user_id, profile in self.user_profiles.items():
                data["user_profiles"][user_id] = {
                    "user_id": profile.user_id,
                    "created_at": profile.created_at.isoformat(),
                    "preferences": profile.preferences,
                    "search_patterns": profile.search_patterns,
                    "language_preferences": profile.language_preferences,
                    "common_paths": profile.common_paths,
                    "session_count": profile.session_count,
                    "total_search_time": profile.total_search_time,
                    "successful_searches": profile.successful_searches,
                    "failed_searches": profile.failed_searches,
                }

            # Convert correlation graph
            for pattern, correlations in self.correlation_graph.items():
                data["correlation_graph"][pattern] = list(correlations)

            with open(self.session_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            # Log error but don't fail the operation
            print(f"Warning: Failed to save sessions: {e}")

    def _load_sessions(self) -> None:
        """Load sessions from persistent storage."""
        try:
            if not Path(self.session_file).exists():
                return

            with open(self.session_file) as f:
                data = json.load(f)

            # Load sessions
            sessions_data = data.get("sessions", {})
            for session_id, session_data in sessions_data.items():
                # Convert back from serializable format
                session_data["current_files"] = set(session_data.get("current_files", []))
                session_data["patterns_discovered"] = set(
                    session_data.get("patterns_discovered", [])
                )
                session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                session_data["last_accessed"] = datetime.fromisoformat(
                    session_data["last_accessed"]
                )
                session_data["priority"] = SessionPriority(session_data.get("priority", "normal"))
                session_data["intent"] = SearchIntent(session_data.get("intent", "unknown"))

                session = EnhancedSearchSession(**session_data)
                self.sessions[session_id] = session

            # Load user profiles
            profiles_data = data.get("user_profiles", {})
            for user_id, profile_data in profiles_data.items():
                profile = UserProfile(user_id)
                profile.created_at = datetime.fromisoformat(profile_data["created_at"])
                profile.preferences = profile_data.get("preferences", {})
                profile.search_patterns = profile_data.get("search_patterns", {})
                profile.language_preferences = profile_data.get("language_preferences", {})
                profile.common_paths = profile_data.get("common_paths", {})
                profile.session_count = profile_data.get("session_count", 0)
                profile.total_search_time = profile_data.get("total_search_time", 0.0)
                profile.successful_searches = profile_data.get("successful_searches", 0)
                profile.failed_searches = profile_data.get("failed_searches", 0)

                self.user_profiles[user_id] = profile

            # Load correlation graph
            correlations_data = data.get("correlation_graph", {})
            for pattern, correlations in correlations_data.items():
                self.correlation_graph[pattern] = set(correlations)

        except Exception as e:
            # Log error but don't fail the initialization
            print(f"Warning: Failed to load sessions: {e}")


# Global session manager instance
_session_manager: EnhancedSessionManager | None = None


def get_session_manager() -> EnhancedSessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = EnhancedSessionManager()
    return _session_manager


# Simple SessionManager for testing
class SimpleSessionManager:
    """Simplified session manager for testing."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    async def create_session(self, user_id: str, context: dict[str, Any]) -> dict[str, Any]:
        """Create a new session."""
        import hashlib
        import time

        session_id = hashlib.md5(f"{user_id}_{time.time()}".encode()).hexdigest()[:12]
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "context": context,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
        }
        self.sessions[session_id] = session_data
        return session_data

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID."""
        return self.sessions.get(session_id)

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update session data."""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
            self.sessions[session_id]["last_accessed"] = datetime.now()


# Use simplified manager for testing, enhanced for production
SessionManager = SimpleSessionManager
