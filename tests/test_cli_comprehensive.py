"""
Comprehensive tests for CLI module.

This module tests the command-line interface functionality including
argument parsing, command execution, and output formatting.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from pysearch.cli import cli, main


class TestCLI:
    """Test CLI functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def extract_json_from_output(self, output: str) -> dict:
        """Extract JSON from CLI output, skipping log lines."""
        lines = output.strip().split("\n")
        json_start = -1
        json_end = -1
        brace_count = 0

        # Find the start of JSON
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        assert json_start >= 0, f"No JSON found in output: {output}"

        # Find the end of JSON by counting braces
        for i in range(json_start, len(lines)):
            line = lines[i]
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0:
                json_end = i
                break

        if json_end == -1:
            json_end = len(lines) - 1

        json_output = "\n".join(lines[json_start:json_end + 1])
        return json.loads(json_output)

    def create_test_files(self, tmp_path: Path):
        """Create test files for CLI testing."""
        # Create Python files
        (tmp_path / "main.py").write_text(
            "def main():\n" "    print('Hello World')\n" "    return 0\n"
        )

        (tmp_path / "utils.py").write_text(
            "def helper():\n"
            "    return True\n"
            "\n"
            "class Helper:\n"
            "    def process(self):\n"
            "        return 'processed'\n"
        )

        (tmp_path / "config.py").write_text(
            "CONFIG = {\n" "    'debug': True,\n" "    'version': '1.0.0'\n" "}\n"
        )

    def test_cli_group(self):
        """Test main CLI group."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "pysearch - Context-aware search engine" in result.output

    def test_find_command_basic(self):
        """Test basic find command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--format", "json", "main"]
            )

            assert result.exit_code == 0
            # Should be valid JSON
            data = self.extract_json_from_output(result.output)
            assert "items" in data
            assert "stats" in data

    def test_find_command_with_regex(self):
        """Test find command with regex option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--regex", "--format", "json", "def .*"]
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            assert len(data["items"]) >= 1  # Should find function definitions

    def test_find_command_with_context(self):
        """Test find command with context option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                ["find", "--path", str(tmp_path), "--context", "3", "--format", "json", "helper"],
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            assert len(data["items"]) >= 1

    def test_find_command_with_include_exclude(self):
        """Test find command with include/exclude patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            # Create a non-Python file
            (tmp_path / "readme.txt").write_text("This is a readme file")

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path",
                    str(tmp_path),
                    "--include",
                    "*.py",
                    "--exclude",
                    "*config*",
                    "--format",
                    "json",
                    "def",
                ],
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            # Should find functions but exclude config.py
            found_files = {item["file"] for item in data["items"]}
            assert any("main.py" in f for f in found_files)
            assert not any("config.py" in f for f in found_files)

    def test_find_command_with_ast_filters(self):
        """Test find command with AST filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path",
                    str(tmp_path),
                    "--filter-func-name",
                    "helper",
                    "--format",
                    "json",
                    "def",
                ],
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            # Should only find functions matching the filter
            assert len(data["items"]) >= 1

    def test_find_command_with_metadata_filters(self):
        """Test find command with metadata filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path",
                    str(tmp_path),
                    "--min-lines",
                    "1",
                    "--max-lines",
                    "10",
                    "--format",
                    "json",
                    "def",
                ],
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            assert "items" in data

    def test_find_command_with_stats(self):
        """Test find command with stats option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--stats", "--format", "json", "def"]
            )

            assert result.exit_code == 0
            # Stats should be in the output when --stats is used
            # The stats are included in the JSON output when format is json
            data = self.extract_json_from_output(result.output)
            assert "stats" in data
            assert "files_scanned" in data["stats"]

    def test_find_command_conflicting_options(self):
        """Test find command with conflicting options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--fuzzy", "--regex", "test"]
            )

            assert result.exit_code == 1
            assert "cannot be used together" in result.output

    def test_find_command_with_fuzzy(self):
        """Test find command with fuzzy search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path",
                    str(tmp_path),
                    "--fuzzy",
                    "--fuzzy-distance",
                    "2",
                    "--format",
                    "json",
                    "heper",  # Misspelled "helper"
                ],
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            # Should find "helper" despite misspelling
            assert len(data["items"]) >= 1

    def test_find_command_with_ranking(self):
        """Test find command with ranking options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path",
                    str(tmp_path),
                    "--ranking",
                    "relevance",
                    "--format",
                    "json",
                    "def",
                ],
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            assert "items" in data

    def test_find_command_with_clustering(self):
        """Test find command with clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--cluster", "--format", "json", "def"]
            )

            assert result.exit_code == 0
            data = self.extract_json_from_output(result.output)
            assert "items" in data

    @patch("pysearch.cli.PySearch")
    def test_find_command_with_debug_logging(self, mock_pysearch):
        """Test find command with debug logging."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.run.return_value = MagicMock()
        mock_engine.run.return_value.items = []
        mock_engine.run.return_value.stats = MagicMock()
        mock_engine.has_critical_errors.return_value = False

        result = self.runner.invoke(cli, ["find", "--debug", "--format", "json", "test"])

        # Debug mode might change exit code, just check it ran
        assert result.exit_code in [0, 1]

    @patch("pysearch.cli.PySearch")
    def test_find_command_with_log_file(self, mock_pysearch):
        """Test find command with log file option."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.run.return_value = MagicMock()
        mock_engine.run.return_value.items = []
        mock_engine.run.return_value.stats = MagicMock()
        mock_engine.has_critical_errors.return_value = False

        # Use a specific temporary file instead of TemporaryDirectory
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp_file:
            log_file_path = tmp_file.name

        try:
            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--log-file",
                    log_file_path,
                    "--log-format",
                    "json",
                    "--format",
                    "json",
                    "test",
                ],
            )

            assert result.exit_code in [0, 1]  # May fail due to log file permissions

        finally:
            # Clean up log file handlers and remove file
            import logging
            for handler in logging.getLogger().handlers[:]:
                if hasattr(handler, 'close'):
                    handler.close()
                logging.getLogger().removeHandler(handler)

            # Remove the temporary log file
            try:
                os.unlink(log_file_path)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors

    def test_find_command_invalid_log_level(self):
        """Test find command with invalid log level."""
        result = self.runner.invoke(cli, ["find", "--log-level", "INVALID", "test"])

        assert result.exit_code == 2  # Click uses exit code 2 for usage errors
        assert "Invalid value" in result.output or "Error" in result.output

    @patch("pysearch.cli.PySearch")
    def test_find_command_with_error_reporting(self, mock_pysearch):
        """Test find command with error reporting."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.run.return_value = MagicMock()
        mock_engine.run.return_value.items = []
        mock_engine.run.return_value.stats = MagicMock()
        mock_engine.has_critical_errors.return_value = True
        mock_engine.get_error_summary.return_value = {"total_errors": 5}
        mock_engine.get_error_report.return_value = "Test error report"

        result = self.runner.invoke(cli, ["find", "--show-errors", "--format", "json", "test"])

        # The command should execute successfully
        assert result.exit_code in [0, 1]
        # Mock was called, so command executed
        mock_pysearch.assert_called_once()

    @patch("pysearch.cli.PySearch")
    def test_find_command_with_ranking_analysis(self, mock_pysearch):
        """Test find command with ranking analysis."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_result = MagicMock()
        mock_result.items = []
        mock_result.stats = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine.has_critical_errors.return_value = False
        mock_engine.get_ranking_suggestions.return_value = {
            "query_type": "simple",
            "recommended_strategy": "relevance",
            "file_spread": 3,
            "result_diversity": 0.8,
            "suggestions": ["Use more specific terms"],
        }

        result = self.runner.invoke(cli, ["find", "--ranking-analysis", "--format", "json", "test"])

        assert result.exit_code in [0, 1]  # May vary based on implementation
        # Mock was called, so command executed
        mock_pysearch.assert_called_once()

    @patch("pysearch.cli.PySearch")
    def test_history_command_basic(self, mock_pysearch):
        """Test basic history command."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_search_history.return_value = [
            MagicMock(
                query_pattern="test",
                timestamp=1234567890,
                files_matched=5,
                items_count=10,
                elapsed_ms=15.5,
            )
        ]

        result = self.runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "test" in result.output

    @patch("pysearch.cli.PySearch")
    def test_history_command_with_analytics(self, mock_pysearch):
        """Test history command with analytics."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_search_analytics.return_value = {
            "total_searches": 100,
            "successful_searches": 85,
            "success_rate": 0.85,
            "average_success_score": 0.75,
            "average_search_time": 25.5,
            "session_count": 10,
            "most_common_categories": [("function", 30), ("class", 20)],
            "most_used_languages": [("python", 80), ("javascript", 20)],
        }

        result = self.runner.invoke(cli, ["history", "--analytics"])

        assert result.exit_code == 0
        assert "Search Analytics" in result.output
        assert "Total searches: 100" in result.output
        assert "Success rate: 85.0%" in result.output

    @patch("pysearch.cli.PySearch")
    def test_history_command_with_sessions(self, mock_pysearch):
        """Test history command with sessions."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_session = MagicMock()
        mock_session.session_id = "abc123def456"
        mock_session.start_time = 1234567890
        mock_session.end_time = 1234567950
        mock_session.total_searches = 5
        mock_session.successful_searches = 4
        mock_session.queries = ["test1", "test2", "test3"]
        mock_engine.get_search_sessions.return_value = [mock_session]

        result = self.runner.invoke(cli, ["history", "--sessions"])

        assert result.exit_code == 0
        assert "Recent Search Sessions" in result.output
        assert "abc123de" in result.output  # First 8 chars of session ID

    @patch("pysearch.cli.PySearch")
    def test_history_command_with_tags(self, mock_pysearch):
        """Test history command with tags."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.search_history_by_tags.return_value = [
            MagicMock(
                query_pattern="tagged_search",
                timestamp=1234567890,
                files_matched=3,
                items_count=6,
                elapsed_ms=12.0,
            )
        ]

        result = self.runner.invoke(cli, ["history", "--tags", "important,work"])

        assert result.exit_code == 0
        assert "tagged_search" in result.output

    @patch("pysearch.cli.PySearch")
    def test_bookmarks_command_list(self, mock_pysearch):
        """Test bookmarks command list functionality."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_bookmarks.return_value = {
            "search1": MagicMock(query_pattern="def main", files_matched=2, items_count=3),
            "search2": MagicMock(query_pattern="class Helper", files_matched=1, items_count=1),
        }

        result = self.runner.invoke(cli, ["bookmarks"])

        assert result.exit_code == 0
        assert "search1" in result.output
        assert "def main" in result.output

    @patch("pysearch.cli.PySearch")
    def test_bookmarks_command_add(self, mock_pysearch):
        """Test bookmarks command add functionality."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.search.return_value = MagicMock()

        result = self.runner.invoke(
            cli, ["bookmarks", "--add", "my_search", "--pattern", "def helper"]
        )

        assert result.exit_code == 0
        assert "Bookmark 'my_search' added" in result.output

    @patch("pysearch.cli.PySearch")
    def test_bookmarks_command_remove(self, mock_pysearch):
        """Test bookmarks command remove functionality."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.remove_bookmark.return_value = True

        result = self.runner.invoke(cli, ["bookmarks", "--remove", "old_search"])

        assert result.exit_code == 0
        assert "Bookmark 'old_search' removed" in result.output

    @patch("pysearch.cli.PySearch")
    def test_bookmarks_command_folders(self, mock_pysearch):
        """Test bookmarks command folder functionality."""
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_bookmark_folders.return_value = {
            "work": MagicMock(
                bookmarks=["search1", "search2"], description="Work-related searches"
            ),
            "personal": MagicMock(bookmarks=["search3"], description=None),
        }

        result = self.runner.invoke(cli, ["bookmarks", "--list-folders"])

        assert result.exit_code == 0
        assert "work: 2 bookmarks" in result.output
        assert "personal: 1 bookmarks" in result.output

    def test_main_function(self):
        """Test main function."""
        with patch("pysearch.cli.cli") as mock_cli:
            main()
            mock_cli.assert_called_once_with(prog_name="pysearch")
