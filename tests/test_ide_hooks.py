"""
Tests for IDE hooks module.

This module tests the IDE integration functionality that provides
a simplified interface for IDE/editor integration.
"""

import tempfile
from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.ide_hooks import ide_query
from pysearch.types import OutputFormat, Query


class TestIDEHooks:
    """Test IDE integration hooks."""

    def test_ide_query_basic(self):
        """Test basic IDE query functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test file
            test_file = tmp_path / "test.py"
            test_file.write_text("def hello():\n    print('Hello World')\n    return True")

            # Setup search engine
            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Create query
            query = Query(pattern="hello", output=OutputFormat.JSON)

            # Execute IDE query
            result = ide_query(engine, query)

            # Verify result structure
            assert isinstance(result, dict)
            assert "items" in result
            assert "stats" in result
            assert isinstance(result["items"], list)
            assert isinstance(result["stats"], dict)

            # Verify items structure
            assert len(result["items"]) >= 1
            item = result["items"][0]
            assert "file" in item
            assert "start_line" in item
            assert "end_line" in item
            assert "lines" in item
            assert "spans" in item

            # Verify content
            assert str(test_file) in item["file"]
            assert item["start_line"] >= 1
            assert item["end_line"] >= item["start_line"]
            assert isinstance(item["lines"], list)
            assert isinstance(item["spans"], list)

    def test_ide_query_with_matches(self):
        """Test IDE query with specific match spans."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test file with multiple matches
            test_file = tmp_path / "multi.py"
            test_file.write_text(
                "def process_data():\n"
                "    data = get_data()\n"
                "    return process(data)\n"
                "\n"
                "def process_file():\n"
                "    return process_content()\n"
            )

            # Setup search engine
            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Create query for "process"
            query = Query(pattern="process", output=OutputFormat.JSON)

            # Execute IDE query
            result = ide_query(engine, query)

            # Should find multiple matches
            assert len(result["items"]) >= 2

            # Verify spans are properly formatted
            for item in result["items"]:
                assert isinstance(item["spans"], list)
                for span in item["spans"]:
                    assert isinstance(span, tuple)
                    assert len(span) == 2
                    line_idx, (start, end) = span
                    assert isinstance(line_idx, int)
                    assert isinstance(start, int)
                    assert isinstance(end, int)
                    assert start <= end

    def test_ide_query_no_matches(self):
        """Test IDE query when no matches are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test file
            test_file = tmp_path / "empty.py"
            test_file.write_text("# Empty file\npass\n")

            # Setup search engine
            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Create query for non-existent pattern
            query = Query(pattern="nonexistent_pattern_xyz", output=OutputFormat.JSON)

            # Execute IDE query
            result = ide_query(engine, query)

            # Should return empty results
            assert isinstance(result, dict)
            assert "items" in result
            assert "stats" in result
            assert len(result["items"]) == 0
            assert result["stats"]["items"] == 0

    def test_ide_query_with_context(self):
        """Test IDE query with context lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test file
            test_file = tmp_path / "context.py"
            test_file.write_text(
                "# Header comment\n"
                "import os\n"
                "def main():\n"
                "    print('Hello')\n"
                "    return 0\n"
                "# Footer comment\n"
            )

            # Setup search engine with context
            cfg = SearchConfig(paths=[str(tmp_path)], context=2)
            engine = PySearch(cfg)

            # Create query
            query = Query(pattern="main", output=OutputFormat.JSON)

            # Execute IDE query
            result = ide_query(engine, query)

            # Should find the match with context
            assert len(result["items"]) >= 1
            item = result["items"][0]

            # Should include context lines
            assert len(item["lines"]) > 1  # More than just the matching line
            assert item["start_line"] < 3  # Should start before line 3 (where "main" is)
            assert item["end_line"] > 3  # Should end after line 3

    def test_ide_query_multiple_files(self):
        """Test IDE query across multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create multiple test files
            file1 = tmp_path / "file1.py"
            file1.write_text("def helper():\n    return True\n")

            file2 = tmp_path / "file2.py"
            file2.write_text("def helper_function():\n    return False\n")

            file3 = tmp_path / "file3.py"
            file3.write_text("class helper_class:\n    pass\n")

            # Setup search engine
            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Create query
            query = Query(pattern="helper", output=OutputFormat.JSON)

            # Execute IDE query
            result = ide_query(engine, query)

            # Should find matches in multiple files
            assert len(result["items"]) >= 3

            # Verify files are different
            found_files = {item["file"] for item in result["items"]}
            assert len(found_files) >= 3

    def test_ide_query_stats_structure(self):
        """Test that IDE query returns properly structured stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test files
            for i in range(3):
                test_file = tmp_path / f"test{i}.py"
                test_file.write_text(f"def function_{i}():\n    return {i}\n")

            # Setup search engine
            cfg = SearchConfig(paths=[str(tmp_path)])
            engine = PySearch(cfg)

            # Create query
            query = Query(pattern="function", output=OutputFormat.JSON)

            # Execute IDE query
            result = ide_query(engine, query)

            # Verify stats structure
            stats = result["stats"]
            required_fields = [
                "files_scanned",
                "files_matched",
                "items",
                "elapsed_ms",
                "indexed_files",
            ]

            for field in required_fields:
                assert field in stats
                assert isinstance(stats[field], (int, float))

            # Verify reasonable values
            assert stats["files_scanned"] >= 3
            assert stats["files_matched"] >= 3
            assert stats["items"] >= 3
            assert stats["elapsed_ms"] >= 0
            assert stats["indexed_files"] >= 3
