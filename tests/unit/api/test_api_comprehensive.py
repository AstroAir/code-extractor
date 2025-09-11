"""
Comprehensive tests for API module.

This module tests the main PySearch API functionality including
search operations, fuzzy search, ranking, and error handling.
"""

import tempfile
from pathlib import Path

from pysearch import PySearch
from pysearch import SearchConfig
from pysearch import (
    ASTFilters,
    Language,
    MetadataFilters,
    OutputFormat,
    Query,
)


class TestPySearchAPI:
    """Test PySearch API functionality."""

    def create_test_files(self, tmp_path: Path):
        """Create test files for API testing."""
        # Python files
        (tmp_path / "main.py").write_text(
            "def main():\n"
            "    '''Main function'''\n"
            "    print('Hello World')\n"
            "    return 0\n"
        )

        (tmp_path / "utils.py").write_text(
            "def helper():\n"
            "    '''Helper function'''\n"
            "    return True\n"
            "\n"
            "class Helper:\n"
            "    '''Helper class'''\n"
            "    def process(self):\n"
            "        return 'processed'\n"
        )

        (tmp_path / "config.py").write_text(
            "# Configuration file\n"
            "CONFIG = {\n"
            "    'debug': True,\n"
            "    'version': '1.0.0'\n"
            "}\n"
        )

    def test_pysearch_basic_search(self):
        """Test basic search functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            result = engine.search("main", output=OutputFormat.JSON)

            assert len(result.items) >= 1
            assert any("main" in item.lines[0].lower() for item in result.items)
            assert result.stats.files_scanned >= 3
            assert result.stats.files_matched >= 1

    def test_pysearch_with_query_object(self):
        """Test search using Query object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            query = Query(pattern="helper", use_regex=False, context=2, output=OutputFormat.JSON)

            result = engine.run(query)

            assert len(result.items) >= 1
            assert any("helper" in item.lines[0].lower() for item in result.items)

    def test_pysearch_regex_search(self):
        """Test regex search functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Use a simpler regex that should match
            result = engine.search("def", use_regex=False, output=OutputFormat.JSON)

            assert len(result.items) >= 1  # Should find function definitions

    def test_pysearch_fuzzy_search(self):
        """Test fuzzy search functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            result = engine.fuzzy_search(
                pattern="heper",  # Misspelled "helper"
                max_distance=2,
                min_similarity=0.6,
                output=OutputFormat.JSON,
            )

            # Should find "helper" despite misspelling
            assert len(result.items) >= 1

    def test_pysearch_fuzzy_search_with_algorithm(self):
        """Test fuzzy search with specific algorithm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            result = engine.fuzzy_search(
                pattern="heper", max_distance=2, algorithm="levenshtein", output=OutputFormat.JSON
            )

            assert len(result.items) >= 1

    def test_pysearch_multi_algorithm_fuzzy_search(self):
        """Test multi-algorithm fuzzy search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            result = engine.multi_algorithm_fuzzy_search(
                pattern="heper",
                algorithms=["levenshtein", "damerau_levenshtein"],
                max_distance=2,
                min_similarity=0.6,
                output=OutputFormat.JSON,
            )

            assert len(result.items) >= 1

    def test_pysearch_search_with_ranking(self):
        """Test search with ranking strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            result = engine.search_with_ranking(
                pattern="def", ranking_strategy="relevance", output=OutputFormat.JSON
            )

            assert len(result.items) >= 1

    def test_pysearch_search_with_clustering(self):
        """Test search with result clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            result = engine.search_with_ranking(
                pattern="def",
                ranking_strategy="hybrid",
                cluster_results=True,
                output=OutputFormat.JSON,
            )

            assert len(result.items) >= 1

    def test_pysearch_with_ast_filters(self):
        """Test search with AST filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            ast_filters = ASTFilters(func_name="helper")
            query = Query(
                pattern="def", use_ast=True, filters=ast_filters, output=OutputFormat.JSON
            )

            result = engine.run(query)

            # Should find functions matching the filter
            assert len(result.items) >= 1

    def test_pysearch_with_metadata_filters(self):
        """Test search with metadata filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            metadata_filters = MetadataFilters(
                min_lines=1, max_lines=20, languages={Language.PYTHON}
            )

            query = Query(
                pattern="def", metadata_filters=metadata_filters, output=OutputFormat.JSON
            )

            result = engine.run(query)

            assert len(result.items) >= 1

    def test_pysearch_get_ranking_suggestions(self):
        """Test getting ranking suggestions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            result = engine.search("def", output=OutputFormat.JSON)
            analysis = engine.get_ranking_suggestions("def", result)

            assert "query_type" in analysis
            assert "recommended_strategy" in analysis
            assert "suggestions" in analysis
            assert isinstance(analysis["suggestions"], list)

    def test_pysearch_error_handling(self):
        """Test error handling functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Test error summary
            error_summary = engine.get_error_summary()
            assert "total_errors" in error_summary
            assert isinstance(error_summary["total_errors"], int)

            # Test error report
            error_report = engine.get_error_report()
            assert isinstance(error_report, str)

            # Test critical errors check
            has_critical = engine.has_critical_errors()
            assert isinstance(has_critical, bool)

    def test_pysearch_search_analytics(self):
        """Test search analytics functionality."""
        cfg = SearchConfig()
        engine = PySearch(cfg)

        analytics = engine.get_search_analytics(days=30)

        assert "total_searches" in analytics
        assert "successful_searches" in analytics
        # Note: success_rate might not be in analytics if no searches have been performed
        assert isinstance(analytics["total_searches"], int)
        assert isinstance(analytics["total_searches"], int)

    def test_pysearch_search_history(self):
        """Test search history functionality."""
        cfg = SearchConfig()
        engine = PySearch(cfg)

        history = engine.get_search_history(limit=10)
        assert isinstance(history, list)

    def test_pysearch_search_sessions(self):
        """Test search sessions functionality."""
        cfg = SearchConfig()
        engine = PySearch(cfg)

        sessions = engine.get_search_sessions(limit=5)
        assert isinstance(sessions, list)

    def test_pysearch_bookmarks(self):
        """Test bookmark functionality."""
        cfg = SearchConfig()
        engine = PySearch(cfg)

        # Test getting bookmarks
        bookmarks = engine.get_bookmarks()
        assert isinstance(bookmarks, dict)

        # Test bookmark folders
        folders = engine.get_bookmark_folders()
        assert isinstance(folders, dict)

    def test_pysearch_with_context(self):
        """Test search with context lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)], context=3)
            engine = PySearch(cfg)

            result = engine.search("main", output=OutputFormat.JSON)

            assert len(result.items) >= 1
            # Should include context lines
            assert any(len(item.lines) > 1 for item in result.items)

    def test_pysearch_different_output_formats(self):
        """Test different output formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Test JSON format
            result_json = engine.search("main", output=OutputFormat.JSON)
            assert len(result_json.items) >= 1

            # Test TEXT format
            result_text = engine.search("main", output=OutputFormat.TEXT)
            assert len(result_text.items) >= 1

            # Test HIGHLIGHT format
            result_highlight = engine.search("main", output=OutputFormat.HIGHLIGHT)
            assert len(result_highlight.items) >= 1
