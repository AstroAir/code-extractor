"""Tests for pysearch.core.history.history_core module."""

from __future__ import annotations

import time
from pathlib import Path

from pysearch.core.config import SearchConfig
from pysearch.core.history.history_core import SearchCategory, SearchHistory, SearchHistoryEntry
from pysearch.core.types import Query, SearchItem, SearchResult, SearchStats


def _make_result(
    items_count: int = 3, files_matched: int = 2, elapsed_ms: float = 50.0
) -> SearchResult:
    return SearchResult(
        items=[],
        stats=SearchStats(
            files_scanned=10,
            files_matched=files_matched,
            items=items_count,
            elapsed_ms=elapsed_ms,
            indexed_files=100,
        ),
    )


def _make_result_with_items(tmp_path: Path) -> SearchResult:
    item = SearchItem(
        file=tmp_path / "a.py",
        start_line=1,
        end_line=1,
        lines=["x"],
        match_spans=[(0, (0, 1))],
    )
    stats = SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0)
    return SearchResult(items=[item], stats=stats)


class TestSearchCategory:
    """Tests for SearchCategory enum."""

    def test_values(self):
        assert SearchCategory.FUNCTION == "function"
        assert SearchCategory.CLASS == "class"
        assert SearchCategory.VARIABLE == "variable"
        assert SearchCategory.IMPORT == "import"
        assert SearchCategory.COMMENT == "comment"
        assert SearchCategory.STRING == "string"
        assert SearchCategory.REGEX == "regex"
        assert SearchCategory.GENERAL == "general"


class TestSearchHistoryEntry:
    """Tests for SearchHistoryEntry dataclass."""

    def test_required_fields(self):
        entry = SearchHistoryEntry(
            timestamp=1.0,
            query_pattern="test",
            use_regex=False,
            use_ast=False,
            context=2,
            files_matched=1,
            items_count=3,
            elapsed_ms=10.0,
        )
        assert entry.query_pattern == "test"
        assert entry.items_count == 3

    def test_defaults(self):
        entry = SearchHistoryEntry(
            timestamp=1.0,
            query_pattern="x",
            use_regex=False,
            use_ast=False,
            context=0,
            files_matched=0,
            items_count=0,
            elapsed_ms=0.0,
        )
        assert entry.filters is None
        assert entry.session_id is None
        assert entry.category == SearchCategory.GENERAL
        assert entry.languages is None
        assert entry.paths is None
        assert entry.success_score == 0.0
        assert entry.user_rating is None
        assert entry.tags is None


class TestSearchHistory:
    """Tests for SearchHistory class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        assert h.max_entries == 1000
        assert h._loaded is False

    def test_add_search_and_get_history(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        q = Query(pattern="def main")
        r = _make_result()
        h.add_search(q, r)
        history = h.get_history(limit=1)
        assert len(history) == 1
        assert history[0].query_pattern == "def main"

    def test_get_stats(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        q = Query(pattern="test")
        h.add_search(q, _make_result())
        stats = h.get_stats()
        assert stats["total_searches"] >= 1
        assert "unique_patterns" in stats

    def test_get_frequent_patterns(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        for pattern in ["foo", "foo", "bar"]:
            h.add_search(Query(pattern=pattern), _make_result())
        pats = h.get_frequent_patterns(limit=5)
        assert isinstance(pats, list)
        assert pats[0][0] == "foo"
        assert pats[0][1] == 2

    def test_get_recent_patterns(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="recent1"), _make_result())
        h.add_search(Query(pattern="recent2"), _make_result())
        recent = h.get_recent_patterns(days=1, limit=5)
        assert "recent2" in recent
        assert "recent1" in recent

    def test_clear_history(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="test"), _make_result())
        h.clear_history()
        assert h.get_stats()["total_searches"] == 0

    def test_save_and_load_persistence(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h1 = SearchHistory(cfg)
        h1.add_search(Query(pattern="persist"), _make_result())
        # add_search triggers save internally

        h2 = SearchHistory(cfg)
        # get_history triggers lazy load
        history = h2.get_history()
        assert any(e.query_pattern == "persist" for e in history)

    def test_categorize_function(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern="def process_data")
        assert h._categorize_search(q) == SearchCategory.FUNCTION

    def test_categorize_class(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern="class DataProcessor")
        assert h._categorize_search(q) == SearchCategory.CLASS

    def test_categorize_import(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern="import numpy")
        assert h._categorize_search(q) == SearchCategory.IMPORT

    def test_categorize_variable(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern="data_processor = ")
        assert h._categorize_search(q) == SearchCategory.VARIABLE

    def test_categorize_regex(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern="def.*handler", use_regex=True)
        assert h._categorize_search(q) == SearchCategory.REGEX

    def test_categorize_comment(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern="# TODO")
        assert h._categorize_search(q) == SearchCategory.COMMENT

    def test_categorize_string(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern='"hello world"')
        assert h._categorize_search(q) == SearchCategory.STRING

    def test_categorize_general(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        q = Query(pattern="something")
        assert h._categorize_search(q) == SearchCategory.GENERAL

    def test_success_score_no_results(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        r = _make_result(items_count=0)
        assert h._calculate_success_score(r) == 0.0

    def test_success_score_good_results(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        r = _make_result(items_count=5, files_matched=3, elapsed_ms=50.0)
        score = h._calculate_success_score(r)
        assert score > 0.5

    def test_success_score_too_many_results(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        h = SearchHistory(cfg)
        r = _make_result(items_count=500, files_matched=50, elapsed_ms=200.0)
        score = h._calculate_success_score(r)
        assert 0.0 < score < 0.8

    def test_delegate_bookmark_methods(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        q = Query(pattern="bm_test")
        r = _make_result()
        h.add_bookmark("b1", q, r)
        bookmarks = h.get_bookmarks()
        assert "b1" in bookmarks

        assert h.remove_bookmark("b1") is True
        assert "b1" not in h.get_bookmarks()

    def test_delegate_folder_methods(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        assert h.create_folder("f1", "desc") is True
        assert "f1" in h.get_folders()
        assert h.delete_folder("f1") is True

    def test_delegate_session_methods(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="s1"), _make_result())
        session = h.get_current_session()
        assert session is not None
        sessions = h.get_sessions()
        assert isinstance(sessions, list)

    def test_delegate_analytics_methods(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="a1"), _make_result())
        analytics = h.get_search_analytics(days=1)
        assert "total_searches" in analytics

    def test_delegate_pattern_suggestions(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="process_data"), _make_result())
        h.add_search(Query(pattern="process_info"), _make_result())
        suggestions = h.get_pattern_suggestions("proc", limit=5)
        assert isinstance(suggestions, list)
        assert any("process" in s for s in suggestions)

    def test_delegate_rate_and_tag(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="rateme"), _make_result())
        assert h.rate_search("rateme", 5) is True
        assert h.rate_search("rateme", 6) is False
        assert h.rate_search("nonexistent", 3) is False

        assert h.add_tags_to_search("rateme", {"tag1"}) is True
        tagged = h.search_history_by_tags({"tag1"})
        assert len(tagged) == 1


class TestSearchHistoryEnhanced:
    """Tests for enhanced SearchHistory methods."""

    def test_get_detailed_stats_empty(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        stats = h.get_detailed_stats()
        assert stats["total_searches"] == 0
        assert stats["date_range"] is None

    def test_get_detailed_stats_with_data(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="def main"), _make_result(items_count=5, elapsed_ms=100.0))
        h.add_search(Query(pattern="class Foo"), _make_result(items_count=3, elapsed_ms=200.0))

        stats = h.get_detailed_stats()
        assert stats["total_searches"] == 2
        assert stats["unique_patterns"] == 2
        assert stats["total_elapsed_ms"] == 300.0
        assert stats["average_elapsed_ms"] == 150.0
        assert stats["total_results"] == 8
        assert stats["average_results"] == 4.0
        assert stats["date_range"] is not None
        assert "categories" in stats

    def test_get_history_by_date_range(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="old"), _make_result())
        import time as _time

        _time.sleep(0.05)
        h.add_search(Query(pattern="new"), _make_result())

        entries = list(h._history)
        mid_time = (entries[0].timestamp + entries[1].timestamp) / 2

        recent = h.get_history_by_date_range(start_time=mid_time)
        assert len(recent) == 1
        assert recent[0].query_pattern == "new"

    def test_get_history_by_date_range_with_limit(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        for i in range(5):
            h.add_search(Query(pattern=f"p{i}"), _make_result())

        result = h.get_history_by_date_range(limit=2)
        assert len(result) == 2

    def test_get_history_by_category(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="def foo"), _make_result())
        h.add_search(Query(pattern="class Bar"), _make_result())
        h.add_search(Query(pattern="something"), _make_result())

        funcs = h.get_history_by_category(SearchCategory.FUNCTION)
        assert len(funcs) >= 1
        assert all(e.category == SearchCategory.FUNCTION for e in funcs)

    def test_get_history_by_category_with_limit(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        for i in range(5):
            h.add_search(Query(pattern=f"def func_{i}"), _make_result())

        result = h.get_history_by_category(SearchCategory.FUNCTION, limit=2)
        assert len(result) <= 2

    def test_get_history_by_language(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        # Manually add entries with language info
        h.load()
        h._ensure_managers_loaded()
        from pysearch.core.history.history_core import SearchHistoryEntry

        h._history.append(
            SearchHistoryEntry(
                timestamp=time.time(),
                query_pattern="py_search",
                use_regex=False,
                use_ast=False,
                context=0,
                files_matched=1,
                items_count=1,
                elapsed_ms=10.0,
                languages={"python"},
            )
        )
        h._history.append(
            SearchHistoryEntry(
                timestamp=time.time(),
                query_pattern="js_search",
                use_regex=False,
                use_ast=False,
                context=0,
                files_matched=1,
                items_count=1,
                elapsed_ms=10.0,
                languages={"javascript"},
            )
        )

        py_entries = h.get_history_by_language("python")
        assert len(py_entries) == 1
        assert py_entries[0].query_pattern == "py_search"

    def test_get_history_by_language_case_insensitive(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.load()
        h._history.append(
            SearchHistoryEntry(
                timestamp=time.time(),
                query_pattern="test",
                use_regex=False,
                use_ast=False,
                context=0,
                files_matched=1,
                items_count=1,
                elapsed_ms=10.0,
                languages={"Python"},
            )
        )

        result = h.get_history_by_language("python")
        assert len(result) == 1

    def test_search_history(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="process_data"), _make_result())
        h.add_search(Query(pattern="handle_request"), _make_result())
        h.add_search(Query(pattern="process_info"), _make_result())

        results = h.search_history("process")
        assert len(results) == 2
        assert all("process" in e.query_pattern for e in results)

    def test_search_history_with_limit(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        for i in range(5):
            h.add_search(Query(pattern=f"test_{i}"), _make_result())

        results = h.search_history("test", limit=2)
        assert len(results) == 2

    def test_cleanup_old_history(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.load()
        # Add an old entry
        old_entry = SearchHistoryEntry(
            timestamp=time.time() - 200 * 86400,  # 200 days ago
            query_pattern="old",
            use_regex=False,
            use_ast=False,
            context=0,
            files_matched=0,
            items_count=0,
            elapsed_ms=0.0,
        )
        h._history.append(old_entry)
        h.add_search(Query(pattern="recent"), _make_result())

        removed = h.cleanup_old_history(days=90)
        assert removed == 1
        assert len(h._history) == 1
        assert list(h._history)[0].query_pattern == "recent"

    def test_cleanup_old_history_none_removed(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="recent"), _make_result())

        removed = h.cleanup_old_history(days=1)
        assert removed == 0

    def test_deduplicate_history(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.load()

        now = time.time()
        for i in range(3):
            h._history.append(
                SearchHistoryEntry(
                    timestamp=now + i * 0.5,  # Within 2 seconds
                    query_pattern="duplicate",
                    use_regex=False,
                    use_ast=False,
                    context=0,
                    files_matched=1,
                    items_count=1,
                    elapsed_ms=10.0,
                )
            )
        h._history.append(
            SearchHistoryEntry(
                timestamp=now + 10,
                query_pattern="unique",
                use_regex=False,
                use_ast=False,
                context=0,
                files_matched=1,
                items_count=1,
                elapsed_ms=10.0,
            )
        )

        removed = h.deduplicate_history()
        assert removed == 2
        assert len(h._history) == 2

    def test_deduplicate_no_duplicates(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="a"), _make_result())
        h.add_search(Query(pattern="b"), _make_result())

        removed = h.deduplicate_history()
        assert removed == 0

    def test_deduplicate_single_entry(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        h = SearchHistory(cfg)
        h.add_search(Query(pattern="solo"), _make_result())

        removed = h.deduplicate_history()
        assert removed == 0
