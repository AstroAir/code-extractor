"""
Unit tests for count-only search and per-file limits functionality.

Tests the count-only search mode and max_per_file limit functionality
in the PySearch API.
"""

from pathlib import Path
from unittest.mock import patch

from pysearch.core.api import PySearch
from pysearch.core.config import SearchConfig
from pysearch.core.types import CountResult, Query, SearchItem, SearchResult, SearchStats


class TestCountOnlySearch:
    """Test count-only search functionality."""

    def test_search_count_only_basic(self):
        """Test basic count-only search functionality."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        # Mock the run method to return test data
        mock_result = SearchResult(
            items=[
                SearchItem(
                    file=Path("test1.py"),
                    start_line=1,
                    end_line=1,
                    lines=["def function1():"],
                    match_spans=[(0, (0, 13))],
                ),
                SearchItem(
                    file=Path("test2.py"),
                    start_line=5,
                    end_line=5,
                    lines=["def function2():"],
                    match_spans=[(0, (0, 13))],
                ),
                SearchItem(
                    file=Path("test2.py"),
                    start_line=10,
                    end_line=10,
                    lines=["def function3():"],
                    match_spans=[(0, (0, 13))],
                ),
            ],
            stats=SearchStats(
                files_scanned=2, files_matched=2, items=3, elapsed_ms=15.5, indexed_files=2
            ),
        )

        with patch.object(engine, "run", return_value=mock_result):
            result = engine.search_count_only("def")

            assert isinstance(result, CountResult)
            assert result.total_matches == 3
            assert result.files_matched == 2
            assert result.stats.files_scanned == 2
            assert result.stats.elapsed_ms == 15.5

    def test_search_count_only_with_regex(self):
        """Test count-only search with regex pattern."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        mock_result = SearchResult(
            items=[
                SearchItem(
                    file=Path("test.py"),
                    start_line=1,
                    end_line=1,
                    lines=["def test_handler():"],
                    match_spans=[(0, (0, 17))],
                )
            ],
            stats=SearchStats(
                files_scanned=1, files_matched=1, items=1, elapsed_ms=8.2, indexed_files=1
            ),
        )

        with patch.object(engine, "run", return_value=mock_result):
            result = engine.search_count_only(r"def .*_handler", regex=True)

            assert result.total_matches == 1
            assert result.files_matched == 1

    def test_search_count_only_with_boolean_query(self):
        """Test count-only search with boolean query."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        mock_result = SearchResult(
            items=[
                SearchItem(
                    file=Path("handler.py"),
                    start_line=1,
                    end_line=1,
                    lines=["async def request_handler():"],
                    match_spans=[(0, (0, 26))],
                )
            ],
            stats=SearchStats(
                files_scanned=3, files_matched=1, items=1, elapsed_ms=12.8, indexed_files=3
            ),
        )

        with patch.object(engine, "run", return_value=mock_result):
            result = engine.search_count_only("async AND handler NOT test", use_boolean=True)

            assert result.total_matches == 1
            assert result.files_matched == 1

    def test_search_count_only_no_matches(self):
        """Test count-only search when no matches are found."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        mock_result = SearchResult(
            items=[],
            stats=SearchStats(
                files_scanned=5, files_matched=0, items=0, elapsed_ms=22.1, indexed_files=5
            ),
        )

        with patch.object(engine, "run", return_value=mock_result):
            result = engine.search_count_only("nonexistent_pattern")

            assert result.total_matches == 0
            assert result.files_matched == 0
            assert result.stats.files_scanned == 5

    def test_search_count_only_query_construction(self):
        """Test that count-only search constructs the Query correctly."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        # Mock run method to capture the query
        captured_query = None

        def mock_run(query):
            nonlocal captured_query
            captured_query = query
            return SearchResult(items=[], stats=SearchStats())

        with patch.object(engine, "run", side_effect=mock_run):
            engine.search_count_only("test_pattern", regex=True, use_boolean=False)

            assert captured_query is not None
            assert captured_query.pattern == "test_pattern"
            assert captured_query.use_regex is True
            assert captured_query.use_boolean is False
            assert captured_query.count_only is True
            assert captured_query.context == 0  # Should be 0 for counting


class TestPerFileLimit:
    """Test per-file limit functionality."""

    def test_max_per_file_limit_applied(self):
        """Test that max_per_file limit is correctly applied."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        # Create test content that would return multiple matches per file

        # Mock _search_file to return multiple items
        def mock_search_file(path, query):
            if query.max_per_file is None:
                return [
                    SearchItem(
                        file=path,
                        start_line=1,
                        end_line=1,
                        lines=["def function1():"],
                        match_spans=[],
                    ),
                    SearchItem(
                        file=path,
                        start_line=4,
                        end_line=4,
                        lines=["def function2():"],
                        match_spans=[],
                    ),
                    SearchItem(
                        file=path,
                        start_line=7,
                        end_line=7,
                        lines=["def function3():"],
                        match_spans=[],
                    ),
                ]
            else:
                items = [
                    SearchItem(
                        file=path,
                        start_line=1,
                        end_line=1,
                        lines=["def function1():"],
                        match_spans=[],
                    ),
                    SearchItem(
                        file=path,
                        start_line=4,
                        end_line=4,
                        lines=["def function2():"],
                        match_spans=[],
                    ),
                    SearchItem(
                        file=path,
                        start_line=7,
                        end_line=7,
                        lines=["def function3():"],
                        match_spans=[],
                    ),
                ]
                return items[: query.max_per_file]

        with patch.object(engine, "_search_file", side_effect=mock_search_file):
            # Test without limit
            query_no_limit = Query(pattern="def", max_per_file=None)
            items_no_limit = engine._search_file(Path("test.py"), query_no_limit)
            assert len(items_no_limit) == 3

            # Test with limit of 2
            query_with_limit = Query(pattern="def", max_per_file=2)
            items_with_limit = engine._search_file(Path("test.py"), query_with_limit)
            assert len(items_with_limit) == 2

            # Test with limit of 1
            query_with_limit_1 = Query(pattern="def", max_per_file=1)
            items_with_limit_1 = engine._search_file(Path("test.py"), query_with_limit_1)
            assert len(items_with_limit_1) == 1

    def test_max_per_file_with_fewer_matches(self):
        """Test max_per_file when file has fewer matches than the limit."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        def mock_search_file(path, query):
            # Return only 2 items regardless of limit
            return [
                SearchItem(
                    file=path, start_line=1, end_line=1, lines=["def function1():"], match_spans=[]
                ),
                SearchItem(
                    file=path, start_line=4, end_line=4, lines=["def function2():"], match_spans=[]
                ),
            ]

        with patch.object(engine, "_search_file", side_effect=mock_search_file):
            # Test with limit higher than actual matches
            query = Query(pattern="def", max_per_file=5)
            items = engine._search_file(Path("test.py"), query)
            assert len(items) == 2  # Should return all available items

    def test_max_per_file_zero_matches(self):
        """Test max_per_file when file has no matches."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        def mock_search_file(path, query):
            return []  # No matches

        with patch.object(engine, "_search_file", side_effect=mock_search_file):
            query = Query(pattern="nonexistent", max_per_file=3)
            items = engine._search_file(Path("test.py"), query)
            assert len(items) == 0

    def test_max_per_file_integration_with_search(self):
        """Test max_per_file integration with full search."""
        config = SearchConfig(paths=["test_data"])
        engine = PySearch(config)

        # Mock the indexer and file operations
        with (
            patch.object(engine.indexer, "scan", return_value=([], [], 1)),
            patch.object(engine.indexer, "save"),
            patch.object(engine.indexer, "iter_all_paths", return_value=[Path("test.py")]),
            patch.object(engine.indexer, "count_indexed", return_value=1),
            patch.object(
                engine,
                "_get_cached_file_content",
                return_value="def func1()\\ndef func2()\\ndef func3()",
            ),
        ):

            # Create a query with max_per_file limit
            query = Query(pattern="def", max_per_file=2, context=0)

            # Mock the actual search_in_file to return 3 matches
            with patch(
                "pysearch.search.matchers.search_in_file",
                return_value=[
                    SearchItem(
                        file=Path("test.py"),
                        start_line=1,
                        end_line=1,
                        lines=["def func1()"],
                        match_spans=[],
                    ),
                    SearchItem(
                        file=Path("test.py"),
                        start_line=2,
                        end_line=2,
                        lines=["def func2()"],
                        match_spans=[],
                    ),
                    SearchItem(
                        file=Path("test.py"),
                        start_line=3,
                        end_line=3,
                        lines=["def func3()"],
                        match_spans=[],
                    ),
                ],
            ):
                result = engine.run(query)

                # Should only return 2 items due to max_per_file limit
                assert len(result.items) == 2


class TestQueryValidation:
    """Test query validation for new features."""

    def test_count_only_flag_in_query(self):
        """Test that count_only flag is properly set in Query."""
        query = Query(pattern="test", count_only=True)
        assert query.count_only is True

        query_default = Query(pattern="test")
        assert query_default.count_only is False

    def test_max_per_file_in_query(self):
        """Test that max_per_file is properly set in Query."""
        query = Query(pattern="test", max_per_file=5)
        assert query.max_per_file == 5

        query_default = Query(pattern="test")
        assert query_default.max_per_file is None

    def test_use_boolean_flag_in_query(self):
        """Test that use_boolean flag is properly set in Query."""
        query = Query(pattern="test AND other", use_boolean=True)
        assert query.use_boolean is True

        query_default = Query(pattern="test")
        assert query_default.use_boolean is False


class TestCountResultType:
    """Test CountResult data type."""

    def test_count_result_creation(self):
        """Test creating CountResult instance."""
        stats = SearchStats(
            files_scanned=10, files_matched=5, items=15, elapsed_ms=123.45, indexed_files=10
        )

        result = CountResult(total_matches=15, files_matched=5, stats=stats)

        assert result.total_matches == 15
        assert result.files_matched == 5
        assert result.stats.files_scanned == 10
        assert result.stats.elapsed_ms == 123.45

    def test_count_result_with_zero_matches(self):
        """Test CountResult with zero matches."""
        stats = SearchStats(
            files_scanned=20, files_matched=0, items=0, elapsed_ms=45.67, indexed_files=20
        )

        result = CountResult(total_matches=0, files_matched=0, stats=stats)

        assert result.total_matches == 0
        assert result.files_matched == 0
        assert result.stats.files_scanned == 20
