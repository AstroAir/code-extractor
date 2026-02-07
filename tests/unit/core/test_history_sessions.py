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
