"""
Tests for mcp/shared/session_manager.py — EnhancedSessionManager,
EnhancedSearchSession, UserProfile, and related classes.

Covers: create_session, get_session, get_or_create_session, record_search,
get_contextual_recommendations, user_profile_management, session_cleanup,
session_analytics, intent inference, contextual suggestions, file context,
search focus, and UserProfile methods.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.shared.session_manager import (
    EnhancedSearchSession,
    EnhancedSessionManager,
    SearchIntent,
    SessionPriority,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_manager():
    """Create a session manager with persistence disabled."""
    return EnhancedSessionManager(persist_sessions=False)


@pytest.fixture
def session_with_user(session_manager):
    """Create a session with a user attached."""
    return session_manager.create_session(user_id="user1")


# ---------------------------------------------------------------------------
# EnhancedSessionManager — session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Tests for session creation, retrieval, and get_or_create."""

    def test_create_session(self, session_manager):
        """create_session returns a session with a valid ID."""
        session = session_manager.create_session(user_id="user1")
        assert session.session_id is not None
        assert session.user_id == "user1"

    def test_create_session_with_explicit_id(self, session_manager):
        """create_session accepts an explicit session_id."""
        session = session_manager.create_session(session_id="my-session", user_id="u1")
        assert session.session_id == "my-session"

    def test_create_session_with_priority(self, session_manager):
        """create_session respects priority parameter."""
        session = session_manager.create_session(user_id="u1", priority=SessionPriority.HIGH)
        assert session.priority == SessionPriority.HIGH

    def test_get_session(self, session_manager):
        """get_session retrieves an existing session by ID."""
        session = session_manager.create_session(user_id="user1")
        retrieved = session_manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_session_not_found(self, session_manager):
        """get_session returns None for unknown session IDs."""
        assert session_manager.get_session("nonexistent") is None

    def test_get_or_create_existing(self, session_manager):
        """get_or_create_session returns existing session when ID matches."""
        s1 = session_manager.create_session(user_id="user1")
        s2 = session_manager.get_or_create_session(session_id=s1.session_id)
        assert s1.session_id == s2.session_id

    def test_get_or_create_new(self, session_manager):
        """get_or_create_session creates new session when ID not found."""
        s = session_manager.get_or_create_session(session_id=None, user_id="u2")
        assert s.session_id is not None


# ---------------------------------------------------------------------------
# EnhancedSessionManager — record_search
# ---------------------------------------------------------------------------


class TestRecordSearch:
    """Tests for session_manager.record_search."""

    def test_record_search_adds_query(self, session_manager, session_with_user):
        """record_search adds an entry to session queries."""
        sid = session_with_user.session_id
        session_manager.record_search(
            session_id=sid,
            query_info={"pattern": "authentication", "type": "text"},
            execution_time=0.5,
            result_count=10,
            success=True,
        )
        updated = session_manager.get_session(sid)
        assert len(updated.queries) == 1

    def test_record_search_updates_analytics(self, session_manager, session_with_user):
        """record_search updates session analytics counters."""
        sid = session_with_user.session_id
        session_manager.record_search(
            session_id=sid,
            query_info={"pattern": "auth", "type": "text"},
            execution_time=0.1,
            result_count=5,
            success=True,
        )
        updated = session_manager.get_session(sid)
        assert updated.total_searches == 1
        assert updated.successful_searches == 1

    def test_record_search_nonexistent_session(self, session_manager):
        """record_search with unknown session_id does not raise."""
        session_manager.record_search(
            session_id="nonexistent",
            query_info={"pattern": "x"},
            execution_time=0.1,
            result_count=0,
        )  # Should not raise


# ---------------------------------------------------------------------------
# EnhancedSessionManager — contextual recommendations
# ---------------------------------------------------------------------------


class TestContextualRecommendations:
    """Tests for get_contextual_recommendations."""

    def test_returns_dict(self, session_manager, session_with_user):
        """get_contextual_recommendations returns a dict."""
        sid = session_with_user.session_id
        session_manager.record_search(
            session_id=sid,
            query_info={"pattern": "auth", "type": "text"},
            execution_time=0.1,
            result_count=5,
        )
        recommendations = session_manager.get_contextual_recommendations(sid)
        assert isinstance(recommendations, dict)

    def test_not_found(self, session_manager):
        """get_contextual_recommendations for unknown session returns error."""
        result = session_manager.get_contextual_recommendations("nonexistent")
        assert "error" in result


# ---------------------------------------------------------------------------
# EnhancedSessionManager — user profiles
# ---------------------------------------------------------------------------


class TestUserProfiles:
    """Tests for user profile management via sessions."""

    def test_profile_created_on_session(self, session_manager):
        """Creating a session with user_id creates a UserProfile."""
        session_manager.create_session(user_id="user1")
        assert "user1" in session_manager.user_profiles
        profile = session_manager.user_profiles["user1"]
        assert isinstance(profile, UserProfile)
        assert profile.user_id == "user1"
        assert profile.session_count >= 1

    def test_profile_session_count_increments(self, session_manager):
        """Creating multiple sessions increments session_count."""
        session_manager.create_session(user_id="u1")
        session_manager.create_session(user_id="u1")
        assert session_manager.user_profiles["u1"].session_count == 2


# ---------------------------------------------------------------------------
# EnhancedSessionManager — session cleanup
# ---------------------------------------------------------------------------


class TestSessionCleanup:
    """Tests for cleanup_old_sessions."""

    def test_cleanup_removes_old_sessions(self, session_manager):
        """Old sessions are cleaned up based on max_age_hours."""
        s1 = session_manager.create_session(session_id="old_s1", user_id="u1")
        s2 = session_manager.create_session(session_id="old_s2", user_id="u2")

        old_time = datetime.now() - timedelta(hours=25)
        s1.last_accessed = old_time
        s2.last_accessed = old_time

        cleaned = session_manager.cleanup_old_sessions(max_age_hours=24)
        assert cleaned >= 2

    def test_cleanup_preserves_recent(self, session_manager):
        """Recent sessions are not cleaned up."""
        session_manager.create_session(session_id="recent", user_id="u1")
        cleaned = session_manager.cleanup_old_sessions(max_age_hours=24)
        assert cleaned == 0
        assert "recent" in session_manager.sessions

    def test_cleanup_preserves_critical(self, session_manager):
        """Critical priority sessions are not cleaned up even if old."""
        s = session_manager.create_session(
            session_id="critical_s", user_id="u1", priority=SessionPriority.CRITICAL
        )
        s.last_accessed = datetime.now() - timedelta(hours=48)
        session_manager.cleanup_old_sessions(max_age_hours=24)
        assert "critical_s" in session_manager.sessions


# ---------------------------------------------------------------------------
# EnhancedSessionManager — analytics
# ---------------------------------------------------------------------------


class TestSessionAnalytics:
    """Tests for get_session_analytics."""

    def test_returns_structure(self, session_manager):
        """get_session_analytics returns expected keys."""
        session_manager.create_session(user_id="u1")
        analytics = session_manager.get_session_analytics()
        assert isinstance(analytics, dict)
        assert "total_sessions" in analytics
        assert "active_sessions" in analytics
        assert "total_users" in analytics
        assert analytics["total_sessions"] >= 1


# ---------------------------------------------------------------------------
# EnhancedSearchSession — direct tests
# ---------------------------------------------------------------------------


class TestEnhancedSearchSession:
    """Tests for EnhancedSearchSession methods."""

    @pytest.fixture
    def session(self):
        return EnhancedSearchSession(session_id="test-session", user_id="u1")

    def test_add_query(self, session):
        """add_query appends to queries list."""
        session.add_query({"pattern": "test", "type": "text"})
        assert len(session.queries) == 1
        assert session.total_searches == 1

    def test_add_successful_search(self, session):
        """add_successful_search updates counters."""
        session.add_successful_search(0.5, 10)
        assert session.successful_searches == 1
        assert session.avg_search_time > 0

    def test_set_search_focus(self, session):
        """set_search_focus updates focus and context."""
        session.set_search_focus("authentication")
        assert session.search_focus == "authentication"
        assert session.context["search_focus"] == "authentication"

    def test_add_file_context(self, session):
        """add_file_context tracks working files."""
        session.add_file_context("/path/to/file.py")
        assert "/path/to/file.py" in session.current_files

    def test_add_file_context_limit(self, session):
        """add_file_context limits the set to 50 files."""
        for i in range(60):
            session.add_file_context(f"/path/file_{i}.py")
        assert len(session.current_files) <= 50

    def test_get_current_context(self, session):
        """get_current_context returns a context snapshot."""
        ctx = session.get_current_context()
        assert ctx["session_id"] == "test-session"
        assert "total_searches" in ctx

    def test_infer_intent_unknown(self, session):
        """infer_intent returns UNKNOWN with no queries."""
        assert session.infer_intent() == SearchIntent.UNKNOWN

    def test_infer_intent_debugging(self, session):
        """infer_intent detects debugging intent."""
        for pattern in ["error handler", "exception trace", "debug log"]:
            session.add_query({"pattern": pattern})
        assert session.infer_intent() == SearchIntent.DEBUGGING

    def test_infer_intent_refactoring(self, session):
        """infer_intent detects refactoring intent."""
        for pattern in ["class rename", "function move", "method extract"]:
            session.add_query({"pattern": pattern})
        assert session.infer_intent() == SearchIntent.REFACTORING

    def test_infer_intent_learning(self, session):
        """infer_intent detects learning intent."""
        for pattern in ["example usage", "how to", "what is"]:
            session.add_query({"pattern": pattern})
        assert session.infer_intent() == SearchIntent.LEARNING

    def test_get_contextual_suggestions(self, session):
        """get_contextual_suggestions returns expected keys."""
        suggestions = session.get_contextual_suggestions()
        assert "suggested_paths" in suggestions
        assert "suggested_patterns" in suggestions
        assert "next_steps" in suggestions


# ---------------------------------------------------------------------------
# UserProfile — direct tests
# ---------------------------------------------------------------------------


class TestUserProfile:
    """Tests for UserProfile methods."""

    @pytest.fixture
    def profile(self):
        return UserProfile("test-user")

    def test_init(self, profile):
        """UserProfile initializes with correct user_id."""
        assert profile.user_id == "test-user"
        assert profile.session_count == 0

    def test_update_preferences(self, profile):
        """update_preferences stores key-value pairs."""
        profile.update_preferences(theme="dark", language="python")
        assert profile.preferences["theme"] == "dark"

    def test_record_search_pattern(self, profile):
        """record_search_pattern increments frequency."""
        profile.record_search_pattern("auth")
        profile.record_search_pattern("auth")
        assert profile.search_patterns["auth"] == 2

    def test_record_language_usage(self, profile):
        """record_language_usage accumulates weights."""
        profile.record_language_usage("python", 1.0)
        profile.record_language_usage("python", 0.5)
        assert profile.language_preferences["python"] == 1.5

    def test_record_path_usage(self, profile):
        """record_path_usage increments count."""
        profile.record_path_usage("./src")
        profile.record_path_usage("./src")
        assert profile.common_paths["./src"] == 2

    def test_get_preferred_languages(self, profile):
        """get_preferred_languages returns top-k languages."""
        profile.record_language_usage("python", 5.0)
        profile.record_language_usage("javascript", 3.0)
        profile.record_language_usage("go", 1.0)
        langs = profile.get_preferred_languages(top_k=2)
        assert langs == ["python", "javascript"]

    def test_get_common_patterns(self, profile):
        """get_common_patterns returns most frequent patterns."""
        for _ in range(5):
            profile.record_search_pattern("auth")
        for _ in range(3):
            profile.record_search_pattern("login")
        patterns = profile.get_common_patterns(top_k=1)
        assert patterns == ["auth"]

    def test_get_suggested_paths(self, profile):
        """get_suggested_paths returns most used paths."""
        for _ in range(3):
            profile.record_path_usage("./src")
        profile.record_path_usage("./tests")
        paths = profile.get_suggested_paths(top_k=1)
        assert paths == ["./src"]
