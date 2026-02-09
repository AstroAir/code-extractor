"""Tests for pysearch.core.api module."""

from __future__ import annotations

from pathlib import Path

from pysearch.core.api import PySearch
from pysearch.core.config import SearchConfig
from pysearch.core.types import OutputFormat, Query, SearchResult, SearchStats


class TestPySearchInit:
    """Tests for PySearch initialization."""

    def test_basic_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.cfg is cfg
        assert engine.history is not None
        assert engine.error_collector is not None
        assert engine.logger is not None

    def test_init_creates_integration_managers(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.hybrid_search is not None
        assert engine.cache_integration is not None
        assert engine.dependency_integration is not None
        assert engine.file_watching is not None
        assert engine.graphrag_integration is not None
        assert engine.indexing_integration is not None
        assert engine.multi_repo_integration is not None
        assert engine.parallel_processing is not None

    def test_init_default_caching_disabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.is_caching_enabled() is False
        assert engine.is_multi_repo_enabled() is False


class TestPySearchSearch:
    """Tests for PySearch search methods."""

    def test_search_basic(self, tmp_path: Path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        result = engine.search("hello")
        assert isinstance(result, SearchResult)
        assert result.stats.files_scanned >= 1

    def test_search_with_regex(self, tmp_path: Path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello_world():\n    pass\n")
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        result = engine.search(r"def \w+", regex=True)
        assert isinstance(result, SearchResult)

    def test_search_no_match(self, tmp_path: Path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        result = engine.search("nonexistent_xyz_pattern")
        assert result.stats.items == 0

    def test_search_count_only(self, tmp_path: Path):
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline1\n")
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        result = engine.search_count_only("line1")
        assert result is not None
        assert result.total_matches >= 1

    def test_run_returns_result(self, tmp_path: Path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def main():\n    pass\n")
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        q = Query(pattern="main")
        result = engine.run(q)
        assert isinstance(result, SearchResult)

    def test_search_output_param(self, tmp_path: Path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        result = engine.search("hello", output=OutputFormat.JSON)
        assert isinstance(result, SearchResult)
        assert result.stats.items >= 1

    def test_search_empty_directory(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        result = engine.search("anything")
        assert result.stats.items == 0


class TestPySearchCaching:
    """Tests for PySearch caching methods."""

    def test_enable_caching(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.enable_caching(backend="memory") is True
        assert engine.is_caching_enabled() is True

    def test_enable_caching_already_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.enable_caching(backend="memory")
        assert engine.enable_caching(backend="memory") is True

    def test_disable_caching(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.enable_caching(backend="memory")
        engine.disable_caching()
        assert engine.is_caching_enabled() is False

    def test_disable_caching_not_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.disable_caching()  # should not raise

    def test_clear_cache(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.enable_caching(backend="memory")
        engine.clear_cache()  # should not raise

    def test_get_cache_stats_not_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        stats = engine.get_cache_stats()
        assert isinstance(stats, dict)
        assert stats.get("enabled") is False

    def test_get_cache_stats_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.enable_caching(backend="memory")
        stats = engine.get_cache_stats()
        assert isinstance(stats, dict)

    def test_invalidate_cache_for_file_not_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.invalidate_cache_for_file("test.py")  # should not raise


class TestPySearchHistory:
    """Tests for PySearch history delegation methods."""

    def test_get_search_history(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("def main(): pass\n")
        engine.search("main")
        history = engine.get_search_history(limit=5)
        assert isinstance(history, list)

    def test_add_and_get_bookmarks(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        q = Query(pattern="test")
        r = SearchResult(items=[], stats=SearchStats())
        engine.add_bookmark("bm1", q, r)
        bms = engine.get_bookmarks()
        assert "bm1" in bms

    def test_remove_bookmark(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        q = Query(pattern="test")
        r = SearchResult(items=[], stats=SearchStats())
        engine.add_bookmark("bm1", q, r)
        assert engine.remove_bookmark("bm1") is True

    def test_get_frequent_patterns(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("foo bar baz\n")
        engine.search("foo")
        engine.search("foo")
        pats = engine.get_frequent_patterns(limit=5)
        assert isinstance(pats, list)

    def test_get_recent_patterns(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("test content\n")
        engine.search("test")
        recent = engine.get_recent_patterns(days=1, limit=5)
        assert isinstance(recent, list)

    def test_get_search_analytics(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("def main(): pass\n")
        engine.search("main")
        analytics = engine.get_search_analytics(days=1)
        assert "total_searches" in analytics

    def test_get_pattern_suggestions(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("process_data\n")
        engine.search("process_data")
        suggestions = engine.get_pattern_suggestions("proc", limit=3)
        assert isinstance(suggestions, list)

    def test_rate_last_search(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("target\n")
        engine.search("target")
        result = engine.rate_last_search("target", 5)
        assert isinstance(result, bool)

    def test_add_search_tags(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("tagged\n")
        engine.search("tagged")
        result = engine.add_search_tags("tagged", ["tag1"])
        assert isinstance(result, bool)

    def test_search_history_by_tags(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("tagged_search\n")
        engine.search("tagged_search")
        engine.add_search_tags("tagged_search", ["t1"])
        result = engine.search_history_by_tags(["t1"])
        assert isinstance(result, list)


class TestPySearchSessions:
    """Tests for PySearch session management."""

    def test_get_current_session(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("session_test\n")
        engine.search("session_test")
        session = engine.get_current_session()
        assert session is not None

    def test_get_search_sessions(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("data\n")
        engine.search("data")
        sessions = engine.get_search_sessions(limit=5)
        assert isinstance(sessions, list)

    def test_end_current_session(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        test_file = tmp_path / "test.py"
        test_file.write_text("data\n")
        engine.search("data")
        engine.end_current_session()


class TestPySearchBookmarkFolders:
    """Tests for PySearch bookmark folder management."""

    def test_create_bookmark_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        assert engine.create_bookmark_folder("work", "Work searches") is True

    def test_get_bookmark_folders(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        engine.create_bookmark_folder("test_folder")
        folders = engine.get_bookmark_folders()
        assert "test_folder" in folders

    def test_delete_bookmark_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        engine.create_bookmark_folder("temp")
        assert engine.delete_bookmark_folder("temp") is True

    def test_add_bookmark_to_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        engine.create_bookmark_folder("f1")
        q = Query(pattern="test")
        r = SearchResult(items=[], stats=SearchStats())
        engine.add_bookmark("bm1", q, r)
        assert engine.add_bookmark_to_folder("bm1", "f1") is True

    def test_remove_bookmark_from_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        engine.create_bookmark_folder("f1")
        q = Query(pattern="test")
        r = SearchResult(items=[], stats=SearchStats())
        engine.add_bookmark("bm1", q, r)
        engine.add_bookmark_to_folder("bm1", "f1")
        assert engine.remove_bookmark_from_folder("bm1", "f1") is True

    def test_get_bookmarks_in_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)
        engine.create_bookmark_folder("f1")
        q = Query(pattern="test")
        r = SearchResult(items=[], stats=SearchStats())
        engine.add_bookmark("bm1", q, r)
        engine.add_bookmark_to_folder("bm1", "f1")
        bms = engine.get_bookmarks_in_folder("f1")
        assert len(bms) == 1


class TestPySearchFileWatching:
    """Tests for PySearch file watching delegation."""

    def test_is_auto_watch_enabled_default(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.is_auto_watch_enabled() is False

    def test_disable_auto_watch(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.disable_auto_watch()  # should not raise

    def test_get_watch_stats(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        stats = engine.get_watch_stats()
        assert isinstance(stats, dict)

    def test_list_watchers(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        watchers = engine.list_watchers()
        assert isinstance(watchers, list)


class TestPySearchMultiRepo:
    """Tests for PySearch multi-repo delegation."""

    def test_is_multi_repo_enabled_default(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.is_multi_repo_enabled() is False

    def test_disable_multi_repo(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.disable_multi_repo()  # should not raise

    def test_list_repositories_not_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.list_repositories() == []

    def test_get_multi_repo_health_not_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.get_multi_repo_health() == {}

    def test_get_multi_repo_stats_not_enabled(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.get_multi_repo_stats() == {}


class TestPySearchErrorHandling:
    """Tests for PySearch error handling methods."""

    def test_get_error_summary(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        summary = engine.get_error_summary()
        assert isinstance(summary, dict)

    def test_get_error_report(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        report = engine.get_error_report()
        assert isinstance(report, str)

    def test_clear_errors(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        engine.clear_errors()  # should not raise

    def test_has_critical_errors(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        assert engine.has_critical_errors() is False

    def test_get_errors_by_category_invalid(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        errors = engine.get_errors_by_category("nonexistent")
        assert errors == []

    def test_get_indexer_stats(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        stats = engine.get_indexer_stats()
        assert isinstance(stats, dict)
