"""Tests for pysearch.core.history.history_sessions module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.history.history_sessions import SearchSession, SessionManager
from pysearch.core.types import Query, SearchItem, SearchResult, SearchStats


def _make_result(items_count: int = 3) -> SearchResult:
    items = []
    if items_count > 0:
        items = [
            SearchItem(file=Path("a.py"), start_line=1, end_line=1, lines=["x"])
            for _ in range(items_count)
        ]
    return SearchResult(
        items=items,
        stats=SearchStats(
            files_scanned=10, files_matched=2, items=items_count, elapsed_ms=50.0,
        ),
    )


class TestSearchSession:
    """Tests for SearchSession dataclass."""

    def test_creation(self):
        s = SearchSession(session_id="abc", start_time=1.0)
        assert s.session_id == "abc"
        assert s.start_time == 1.0
        assert s.end_time is None
        assert s.queries == []
        assert s.total_searches == 0
        assert s.successful_searches == 0
        assert s.primary_paths is None
        assert s.primary_languages is None

    def test_post_init_queries(self):
        s = SearchSession(session_id="x", start_time=1.0)
        assert s.queries is not None
        assert isinstance(s.queries, list)


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        assert mgr._loaded is False
        assert mgr._current_session is None

    def test_get_or_create_session_new(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        session = mgr.get_or_create_session(time.time())
        assert session is not None
        assert session.session_id is not None
        assert session.total_searches == 0

    def test_get_or_create_session_reuse(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        now = time.time()
        s1 = mgr.get_or_create_session(now)
        mgr._last_search_time = now
        s2 = mgr.get_or_create_session(now + 1)
        assert s1.session_id == s2.session_id

    def test_get_or_create_session_timeout(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        now = time.time()
        s1 = mgr.get_or_create_session(now)
        mgr._last_search_time = now
        s2 = mgr.get_or_create_session(now + 3600)  # 1 hour later
        assert s1.session_id != s2.session_id

    def test_update_session(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        session = mgr.get_or_create_session(time.time())
        q = Query(pattern="test")
        r = _make_result(items_count=3)
        mgr.update_session(session, q, r)
        assert session.total_searches == 1
        assert session.successful_searches == 1
        assert "test" in session.queries

    def test_update_session_failed_search(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        session = mgr.get_or_create_session(time.time())
        q = Query(pattern="nothing")
        r = _make_result(items_count=0)
        mgr.update_session(session, q, r)
        assert session.total_searches == 1
        assert session.successful_searches == 0

    def test_get_current_session(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        assert mgr.get_current_session() is None
        mgr.get_or_create_session(time.time())
        assert mgr.get_current_session() is not None

    def test_get_sessions(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        now = time.time()
        mgr.get_or_create_session(now)
        mgr._last_search_time = now
        mgr._current_session = None  # Force new session
        mgr.get_or_create_session(now + 3600)
        sessions = mgr.get_sessions()
        assert len(sessions) == 2

    def test_get_sessions_with_limit(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        now = time.time()
        for i in range(3):
            mgr._current_session = None
            mgr._last_search_time = 0
            mgr.get_or_create_session(now + i * 3600)
        sessions = mgr.get_sessions(limit=2)
        assert len(sessions) == 2

    def test_get_session_by_id(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        session = mgr.get_or_create_session(time.time())
        found = mgr.get_session_by_id(session.session_id)
        assert found is session

    def test_get_session_by_id_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        assert mgr.get_session_by_id("nonexistent") is None

    def test_end_current_session(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr.get_or_create_session(time.time())
        mgr.end_current_session()
        assert mgr._current_session is None

    def test_end_current_session_sets_end_time(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        session = mgr.get_or_create_session(time.time())
        sid = session.session_id
        mgr.end_current_session()
        ended = mgr.get_session_by_id(sid)
        assert ended.end_time is not None

    def test_get_session_analytics_empty(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        analytics = mgr.get_session_analytics(days=30)
        assert analytics["total_sessions"] == 0

    def test_get_session_analytics_with_data(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        session = mgr.get_or_create_session(time.time())
        session.total_searches = 5
        session.successful_searches = 3
        analytics = mgr.get_session_analytics(days=1)
        assert analytics["total_sessions"] == 1
        assert analytics["average_searches_per_session"] == 5.0

    def test_cleanup_old_sessions(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        old_time = time.time() - 200 * 86400  # 200 days ago
        mgr._loaded = True
        mgr._sessions["old"] = SearchSession(session_id="old", start_time=old_time)
        mgr._sessions["new"] = SearchSession(session_id="new", start_time=time.time())
        removed = mgr.cleanup_old_sessions(days=90)
        assert removed == 1
        assert "old" not in mgr._sessions
        assert "new" in mgr._sessions

    def test_persistence(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr1 = SessionManager(cfg)
        session = mgr1.get_or_create_session(time.time())
        session.total_searches = 3
        mgr1.save_sessions()

        mgr2 = SessionManager(cfg)
        mgr2.load()
        sessions = mgr2.get_sessions()
        assert len(sessions) >= 1
        assert any(s.total_searches == 3 for s in sessions)


class TestSessionManagerCompare:
    """Tests for compare_sessions and get_session_summary."""

    def test_compare_sessions(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr._loaded = True

        s1 = SearchSession(
            session_id="s1", start_time=1000.0, end_time=2000.0,
            queries=["foo", "bar"], total_searches=5, successful_searches=4,
            primary_languages={"python"}, primary_paths={"/src"},
        )
        s2 = SearchSession(
            session_id="s2", start_time=3000.0, end_time=4000.0,
            queries=["bar", "baz"], total_searches=3, successful_searches=2,
            primary_languages={"python", "javascript"}, primary_paths={"/src", "/lib"},
        )
        mgr._sessions["s1"] = s1
        mgr._sessions["s2"] = s2

        result = mgr.compare_sessions("s1", "s2")

        assert "session_1" in result
        assert "session_2" in result
        assert result["session_1"]["total_searches"] == 5
        assert result["session_2"]["total_searches"] == 3
        assert "bar" in result["common_queries"]
        assert "foo" in result["only_in_session_1"]
        assert "baz" in result["only_in_session_2"]
        assert "python" in result["common_languages"]
        assert result["search_count_diff"] == -2

    def test_compare_sessions_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr._loaded = True

        result = mgr.compare_sessions("nonexistent1", "nonexistent2")
        assert "error" in result

    def test_compare_sessions_one_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr._loaded = True
        mgr._sessions["s1"] = SearchSession(session_id="s1", start_time=1000.0)

        result = mgr.compare_sessions("s1", "nonexistent")
        assert "error" in result

    def test_get_session_summary(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr._loaded = True

        session = SearchSession(
            session_id="test_session", start_time=1000.0, end_time=2000.0,
            queries=["foo", "bar", "foo"], total_searches=10, successful_searches=8,
            primary_languages={"python", "javascript"}, primary_paths={"/src"},
        )
        mgr._sessions["test_session"] = session

        summary = mgr.get_session_summary("test_session")

        assert summary["session_id"] == "test_session"
        assert summary["total_searches"] == 10
        assert summary["successful_searches"] == 8
        assert summary["failed_searches"] == 2
        assert summary["success_rate"] == 0.8
        assert summary["is_active"] is False
        assert summary["duration_seconds"] == 1000.0
        assert "foo" in summary["repeated_queries"]
        assert "python" in summary["primary_languages"]

    def test_get_session_summary_active(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr._loaded = True

        session = SearchSession(
            session_id="active", start_time=time.time() - 60,
            total_searches=2, successful_searches=1,
        )
        mgr._sessions["active"] = session

        summary = mgr.get_session_summary("active")

        assert summary["is_active"] is True
        assert summary["end_time_iso"] is None

    def test_get_session_summary_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr._loaded = True

        result = mgr.get_session_summary("nonexistent")
        assert "error" in result

    def test_get_session_summary_no_searches(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = SessionManager(cfg)
        mgr._loaded = True

        session = SearchSession(session_id="empty", start_time=1000.0, end_time=1001.0)
        mgr._sessions["empty"] = session

        summary = mgr.get_session_summary("empty")

        assert summary["total_searches"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["unique_queries"] == 0
