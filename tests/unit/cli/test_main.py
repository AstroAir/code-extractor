"""
Tests for pysearch.cli.main module.

Tests are organized by CLI command, matching the structure of main.py:
- TestCliGroup: cli group, version, help
- TestFindCmd: find command
- TestHistoryCmd: history command
- TestBookmarksCmd: bookmarks command
- TestSemanticCmd: semantic command
- TestIndexCmd: index command
- TestDepsCmd: deps command
- TestWatchCmd: watch command
- TestCacheCmd: cache command
- TestConfigCmd: config command
- TestMainFunction: main() entry point
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from pysearch.cli import cli, main


def _extract_json(output: str) -> dict:
    """Extract JSON from CLI output, skipping log lines."""
    lines = output.strip().split("\n")
    json_start = -1
    json_end = -1
    brace_count = 0

    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            json_start = i
            break

    assert json_start >= 0, f"No JSON found in output: {output}"

    for i in range(json_start, len(lines)):
        brace_count += lines[i].count("{") - lines[i].count("}")
        if brace_count == 0:
            json_end = i
            break

    if json_end == -1:
        json_end = len(lines) - 1

    json_output = "\n".join(lines[json_start : json_end + 1])
    return json.loads(json_output)


def _create_test_files(tmp_path: Path):
    """Create test Python files for CLI testing."""
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


# ---------------------------------------------------------------------------
# cli group
# ---------------------------------------------------------------------------


class TestCliGroup:
    """Tests for the top-level cli group."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "pysearch - Context-aware search engine" in result.output

    def test_version(self):
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "pysearch" in result.output


# ---------------------------------------------------------------------------
# find command
# ---------------------------------------------------------------------------


class TestFindCmd:
    """Tests for the find command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["find", "--help"])
        assert result.exit_code == 0
        assert "Execute a search" in result.output

    def test_basic_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--format", "json", "main"]
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert "items" in data
            assert "stats" in data

    def test_regex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--regex", "--format", "json", "def .*"]
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert len(data["items"]) >= 1

    def test_context_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                ["find", "--path", str(tmp_path), "--context", "3", "--format", "json", "helper"],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert len(data["items"]) >= 1

    def test_include_exclude(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)
            (tmp_path / "readme.txt").write_text("This is a readme file")

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path", str(tmp_path),
                    "--include", "*.py",
                    "--exclude", "*config*",
                    "--format", "json",
                    "def",
                ],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            found_files = {item["file"] for item in data["items"]}
            assert any("main.py" in f for f in found_files)
            assert not any("config.py" in f for f in found_files)

    def test_ast_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path", str(tmp_path),
                    "--filter-func-name", "helper",
                    "--format", "json",
                    "def",
                ],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert len(data["items"]) >= 1

    def test_metadata_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path", str(tmp_path),
                    "--min-lines", "1",
                    "--max-lines", "10",
                    "--format", "json",
                    "def",
                ],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert "items" in data

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--stats", "--format", "json", "def"]
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert "stats" in data
            assert "files_scanned" in data["stats"]

    def test_fuzzy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path", str(tmp_path),
                    "--fuzzy",
                    "--fuzzy-distance", "2",
                    "--format", "json",
                    "heper",  # misspelled "helper"
                ],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert len(data["items"]) >= 1

    def test_ranking_relevance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--path", str(tmp_path),
                    "--ranking", "relevance",
                    "--format", "json",
                    "def",
                ],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert "items" in data

    def test_cluster(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--cluster", "--format", "json", "def"]
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert "items" in data

    # -- conflicting options --

    def test_conflict_fuzzy_regex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _create_test_files(tmp_path)

            result = self.runner.invoke(
                cli, ["find", "--path", str(tmp_path), "--fuzzy", "--regex", "test"]
            )
            assert result.exit_code == 1
            assert "cannot be used together" in result.output

    def test_conflict_logic_fuzzy(self):
        result = self.runner.invoke(cli, ["find", "--logic", "--fuzzy", "test"])
        assert result.exit_code == 1
        assert "cannot be used together" in result.output

    def test_conflict_count_highlight(self):
        result = self.runner.invoke(cli, ["find", "--count", "--format", "highlight", "test"])
        assert result.exit_code == 1
        assert "cannot be used" in result.output

    def test_invalid_log_level(self):
        result = self.runner.invoke(cli, ["find", "--log-level", "INVALID", "test"])
        assert result.exit_code == 2  # Click usage error
        assert "Invalid value" in result.output or "Error" in result.output

    # -- mocked tests for features that need engine --

    @patch("pysearch.cli.main.PySearch")
    def test_debug_logging(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.run.return_value = MagicMock(items=[], stats=MagicMock())
        mock_engine.has_critical_errors.return_value = False

        result = self.runner.invoke(cli, ["find", "--debug", "--format", "json", "test"])
        assert result.exit_code in [0, 1]

    @patch("pysearch.cli.main.PySearch")
    def test_log_file(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.run.return_value = MagicMock(items=[], stats=MagicMock())
        mock_engine.has_critical_errors.return_value = False

        import logging

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tmp_file:
            log_file_path = tmp_file.name

        try:
            result = self.runner.invoke(
                cli,
                [
                    "find",
                    "--log-file", log_file_path,
                    "--log-format", "json",
                    "--format", "json",
                    "test",
                ],
            )
            assert result.exit_code in [0, 1]
        finally:
            for handler in logging.getLogger().handlers[:]:
                if hasattr(handler, "close"):
                    handler.close()
                logging.getLogger().removeHandler(handler)
            try:
                os.unlink(log_file_path)
            except (OSError, PermissionError):
                pass

    @patch("pysearch.cli.main.PySearch")
    def test_error_reporting(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.run.return_value = MagicMock(items=[], stats=MagicMock())
        mock_engine.has_critical_errors.return_value = True
        mock_engine.get_error_summary.return_value = {"total_errors": 5}
        mock_engine.get_error_report.return_value = "Test error report"

        result = self.runner.invoke(cli, ["find", "--show-errors", "--format", "json", "test"])
        assert result.exit_code in [0, 1]
        mock_pysearch.assert_called_once()

    @patch("pysearch.cli.main.PySearch")
    def test_ranking_analysis(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_result = MagicMock(items=[], stats=MagicMock())
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
        assert result.exit_code in [0, 1]
        mock_pysearch.assert_called_once()

    @patch("pysearch.cli.main.PySearch")
    def test_count_only_text(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_count = MagicMock()
        mock_count.total_matches = 42
        mock_count.files_matched = 3
        mock_count.stats = MagicMock(files_scanned=10, elapsed_ms=5.0, indexed_files=8)
        mock_engine.search_count_only.return_value = mock_count

        result = self.runner.invoke(cli, ["find", "--count", "test"])
        assert result.exit_code == 0
        assert "Total matches: 42" in result.output
        assert "Files matched: 3" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_count_only_json(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_count = MagicMock()
        mock_count.total_matches = 42
        mock_count.files_matched = 3
        mock_count.stats = MagicMock(files_scanned=10, elapsed_ms=5.0, indexed_files=8)
        mock_engine.search_count_only.return_value = mock_count

        result = self.runner.invoke(cli, ["find", "--count", "--format", "json", "test"])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["total_matches"] == 42

    @patch("pysearch.cli.main.PySearch")
    def test_count_with_stats(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_count = MagicMock()
        mock_count.total_matches = 10
        mock_count.files_matched = 2
        mock_count.stats = MagicMock(files_scanned=5, elapsed_ms=3.0, indexed_files=5)
        mock_engine.search_count_only.return_value = mock_count

        result = self.runner.invoke(cli, ["find", "--count", "--stats", "test"])
        assert result.exit_code == 0
        assert "Files scanned:" in result.output


# ---------------------------------------------------------------------------
# history command
# ---------------------------------------------------------------------------


class TestHistoryCmd:
    """Tests for the history command."""

    def setup_method(self):
        self.runner = CliRunner()

    @patch("pysearch.cli.main.PySearch")
    def test_list(self, mock_pysearch):
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

    @patch("pysearch.cli.main.PySearch")
    def test_analytics(self, mock_pysearch):
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

    @patch("pysearch.cli.main.PySearch")
    def test_analytics_with_days(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_search_analytics.return_value = {
            "total_searches": 50,
            "successful_searches": 40,
            "success_rate": 0.8,
            "average_success_score": 0.7,
            "average_search_time": 20.0,
            "session_count": 5,
            "most_common_categories": [],
            "most_used_languages": [],
        }

        result = self.runner.invoke(cli, ["history", "--analytics", "--days", "14"])
        assert result.exit_code == 0
        assert "Last 14 days" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_sessions(self, mock_pysearch):
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
        assert "abc123de" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_tags_filter(self, mock_pysearch):
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

    @patch("pysearch.cli.main.PySearch")
    def test_clear(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine

        result = self.runner.invoke(cli, ["history", "--clear"])
        assert result.exit_code == 0
        assert "cleared" in result.output
        mock_engine.history.clear_history.assert_called_once()

    @patch("pysearch.cli.main.PySearch")
    def test_frequent(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_frequent_patterns.return_value = [
            ("def main", 15),
            ("class Test", 8),
        ]

        result = self.runner.invoke(cli, ["history", "--frequent"])
        assert result.exit_code == 0
        assert "Most Frequent" in result.output
        assert "def main" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_frequent_empty(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_frequent_patterns.return_value = []

        result = self.runner.invoke(cli, ["history", "--frequent"])
        assert result.exit_code == 0
        assert "No search patterns found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_recent(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_recent_patterns.return_value = ["pattern1", "pattern2"]

        result = self.runner.invoke(cli, ["history", "--recent", "--days", "7"])
        assert result.exit_code == 0
        assert "Recent Search Patterns" in result.output
        assert "Last 7 days" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_recent_empty(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_recent_patterns.return_value = []

        result = self.runner.invoke(cli, ["history", "--recent"])
        assert result.exit_code == 0
        assert "No recent search patterns found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_empty_history(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_search_history.return_value = []

        result = self.runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "No search history found" in result.output


# ---------------------------------------------------------------------------
# bookmarks command
# ---------------------------------------------------------------------------


class TestBookmarksCmd:
    """Tests for the bookmarks command."""

    def setup_method(self):
        self.runner = CliRunner()

    @patch("pysearch.cli.main.PySearch")
    def test_list(self, mock_pysearch):
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

    @patch("pysearch.cli.main.PySearch")
    def test_list_empty(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_bookmarks.return_value = {}

        result = self.runner.invoke(cli, ["bookmarks"])
        assert result.exit_code == 0
        assert "No bookmarks found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_add(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.search.return_value = MagicMock()

        result = self.runner.invoke(
            cli, ["bookmarks", "--add", "my_search", "--pattern", "def helper"]
        )
        assert result.exit_code == 0
        assert "Bookmark 'my_search' added" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_add_without_pattern(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine

        result = self.runner.invoke(cli, ["bookmarks", "--add", "my_bookmark"])
        assert result.exit_code == 1
        assert "--pattern is required" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_add_with_folder(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.search.return_value = MagicMock()

        result = self.runner.invoke(
            cli,
            ["bookmarks", "--add", "my_search", "--pattern", "def helper", "--folder", "work"],
        )
        assert result.exit_code == 0
        assert "Bookmark 'my_search' added" in result.output
        assert "folder 'work'" in result.output
        mock_engine.add_bookmark_to_folder.assert_called_once_with("my_search", "work")

    @patch("pysearch.cli.main.PySearch")
    def test_remove(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.remove_bookmark.return_value = True

        result = self.runner.invoke(cli, ["bookmarks", "--remove", "old_search"])
        assert result.exit_code == 0
        assert "Bookmark 'old_search' removed" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_remove_not_found(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.remove_bookmark.return_value = False

        result = self.runner.invoke(cli, ["bookmarks", "--remove", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_list_folders(self, mock_pysearch):
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

    @patch("pysearch.cli.main.PySearch")
    def test_list_folders_empty(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_bookmark_folders.return_value = {}

        result = self.runner.invoke(cli, ["bookmarks", "--list-folders"])
        assert result.exit_code == 0
        assert "No bookmark folders found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_create_folder(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.create_bookmark_folder.return_value = True

        result = self.runner.invoke(
            cli,
            ["bookmarks", "--create-folder", "work", "--description", "Work-related searches"],
        )
        assert result.exit_code == 0
        assert "created" in result.output
        mock_engine.create_bookmark_folder.assert_called_once_with("work", "Work-related searches")

    @patch("pysearch.cli.main.PySearch")
    def test_create_folder_already_exists(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.create_bookmark_folder.return_value = False

        result = self.runner.invoke(cli, ["bookmarks", "--create-folder", "work"])
        assert result.exit_code == 0
        assert "already exists" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_delete_folder(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.delete_bookmark_folder.return_value = True

        result = self.runner.invoke(cli, ["bookmarks", "--delete-folder", "old_folder"])
        assert result.exit_code == 0
        assert "deleted" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_delete_folder_not_found(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.delete_bookmark_folder.return_value = False

        result = self.runner.invoke(cli, ["bookmarks", "--delete-folder", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_list_bookmarks_in_folder(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_bookmarks_in_folder.return_value = [
            MagicMock(query_pattern="def main", files_matched=2, items_count=3),
        ]

        result = self.runner.invoke(cli, ["bookmarks", "--folder", "work"])
        assert result.exit_code == 0
        assert "Bookmarks in folder 'work'" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_list_bookmarks_in_folder_empty(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_bookmarks_in_folder.return_value = []

        result = self.runner.invoke(cli, ["bookmarks", "--folder", "empty_folder"])
        assert result.exit_code == 0
        assert "No bookmarks found in folder" in result.output


# ---------------------------------------------------------------------------
# semantic command
# ---------------------------------------------------------------------------


class TestSemanticCmd:
    """Tests for the semantic command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["semantic", "--help"])
        assert result.exit_code == 0
        assert "语义搜索" in result.output or "semantic" in result.output.lower()

    @patch("pysearch.cli.main.PySearch")
    def test_basic(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_result = MagicMock()
        mock_result.items = []
        mock_result.stats = MagicMock(files_matched=0, items=0, elapsed_ms=5.0)
        mock_engine.semantic_search.return_value = mock_result

        result = self.runner.invoke(
            cli, ["semantic", "--path", ".", "--format", "json", "database connection"]
        )
        assert result.exit_code in [0, 1]
        mock_engine.semantic_search.assert_called_once()

    @patch("pysearch.cli.main.PySearch")
    def test_error_handling(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.semantic_search.side_effect = RuntimeError("model not loaded")

        result = self.runner.invoke(cli, ["semantic", "database connection"])
        assert result.exit_code == 1
        assert "Error during semantic search" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_max_results_truncation(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_result = MagicMock()
        mock_result.items = [MagicMock() for _ in range(200)]
        mock_result.stats = MagicMock(files_matched=5, items=200, elapsed_ms=50.0)
        mock_engine.semantic_search.return_value = mock_result

        result = self.runner.invoke(
            cli, ["semantic", "--max-results", "50", "--format", "json", "query"]
        )
        assert result.exit_code in [0, 1]
        assert len(mock_result.items) == 50


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


class TestIndexCmd:
    """Tests for the index command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "索引" in result.output or "index" in result.output.lower()

    @patch("pysearch.cli.main.PySearch")
    def test_stats(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_indexer_stats.return_value = {
            "total_files": 100,
            "cached_files": 80,
        }

        result = self.runner.invoke(cli, ["index", "--stats"])
        assert result.exit_code == 0
        assert "Index Statistics" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_cleanup(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.cleanup_old_cache_entries.return_value = 5

        result = self.runner.invoke(cli, ["index", "--cleanup", "30"])
        assert result.exit_code == 0
        assert "Cleaned up 5" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_rebuild(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.indexer.scan.return_value = (["file1.py"], [], 10)

        result = self.runner.invoke(cli, ["index", "--rebuild"])
        assert result.exit_code == 0
        assert "rebuilt" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_default_scan(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.indexer.scan.return_value = (["file1.py"], [], 20)

        result = self.runner.invoke(cli, ["index"])
        assert result.exit_code == 0
        assert "Index status" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_default_scan_error(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.indexer.scan.side_effect = RuntimeError("scan failed")

        result = self.runner.invoke(cli, ["index"])
        assert result.exit_code == 1
        assert "Error scanning index" in result.output


# ---------------------------------------------------------------------------
# deps command
# ---------------------------------------------------------------------------


class TestDepsCmd:
    """Tests for the deps command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["deps", "--help"])
        assert result.exit_code == 0

    @patch("pysearch.cli.main.PySearch")
    def test_metrics(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_dependency_metrics.return_value = {
            "total_modules": 10,
            "circular_deps": 0,
        }

        result = self.runner.invoke(cli, ["deps", "--metrics"])
        assert result.exit_code == 0
        assert "Dependency Metrics" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_metrics_json(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_dependency_metrics.return_value = {
            "total_modules": 10,
            "circular_deps": 0,
        }

        result = self.runner.invoke(cli, ["deps", "--metrics", "--format", "json"])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["total_modules"] == 10

    @patch("pysearch.cli.main.PySearch")
    def test_metrics_error(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_dependency_metrics.side_effect = RuntimeError("fail")

        result = self.runner.invoke(cli, ["deps", "--metrics"])
        assert result.exit_code == 1
        assert "Error calculating metrics" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_impact(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.find_dependency_impact.return_value = {
            "total_affected_modules": 3,
            "impact_score": 0.5,
            "direct_dependents": ["mod_a", "mod_b"],
        }

        result = self.runner.invoke(cli, ["deps", "--impact", "src.core.api"])
        assert result.exit_code == 0
        assert "Impact Analysis" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_impact_json(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.find_dependency_impact.return_value = {
            "total_affected_modules": 3,
            "impact_score": 0.5,
        }

        result = self.runner.invoke(
            cli, ["deps", "--impact", "src.core.api", "--format", "json"]
        )
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["total_affected_modules"] == 3

    @patch("pysearch.cli.main.PySearch")
    def test_impact_error(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.find_dependency_impact.side_effect = RuntimeError("fail")

        result = self.runner.invoke(cli, ["deps", "--impact", "bad.module"])
        assert result.exit_code == 1
        assert "Error analyzing impact" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_suggest(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.suggest_refactoring_opportunities.return_value = [
            {
                "priority": "high",
                "type": "circular_dependency",
                "description": "Break circular dep between A and B",
                "rationale": "Reduces coupling",
            }
        ]

        result = self.runner.invoke(cli, ["deps", "--suggest"])
        assert result.exit_code == 0
        assert "Refactoring Suggestions" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_suggest_empty(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.suggest_refactoring_opportunities.return_value = []

        result = self.runner.invoke(cli, ["deps", "--suggest"])
        assert result.exit_code == 0
        assert "No refactoring suggestions found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_suggest_error(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.suggest_refactoring_opportunities.side_effect = RuntimeError("fail")

        result = self.runner.invoke(cli, ["deps", "--suggest"])
        assert result.exit_code == 1
        assert "Error generating suggestions" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_default_analysis(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_graph = MagicMock()
        mock_graph.nodes = ["a", "b", "c"]
        mock_graph.edges = [("a", "b")]
        mock_engine.analyze_dependencies.return_value = mock_graph

        result = self.runner.invoke(cli, ["deps"])
        assert result.exit_code == 0
        assert "Dependency Analysis" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_default_no_deps(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.analyze_dependencies.return_value = None

        result = self.runner.invoke(cli, ["deps"])
        assert result.exit_code == 0
        assert "No dependencies found" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_default_error(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.analyze_dependencies.side_effect = RuntimeError("fail")

        result = self.runner.invoke(cli, ["deps"])
        assert result.exit_code == 1
        assert "Error analyzing dependencies" in result.output


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------


class TestWatchCmd:
    """Tests for the watch command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0

    @patch("pysearch.cli.main.PySearch")
    def test_status_disabled(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.is_auto_watch_enabled.return_value = False

        result = self.runner.invoke(cli, ["watch", "--status"])
        assert result.exit_code == 0
        assert "DISABLED" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_status_enabled(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.is_auto_watch_enabled.return_value = True
        mock_engine.list_watchers.return_value = ["watcher1"]
        mock_engine.get_watch_stats.return_value = {
            "watcher1": {"events": 10}
        }

        result = self.runner.invoke(cli, ["watch", "--status"])
        assert result.exit_code == 0
        assert "ENABLED" in result.output
        assert "watcher1" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_enable(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.enable_auto_watch.return_value = True

        result = self.runner.invoke(cli, ["watch", "--enable"])
        assert result.exit_code == 0
        assert "enabled" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_enable_failure(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.enable_auto_watch.return_value = False

        result = self.runner.invoke(cli, ["watch", "--enable"])
        assert result.exit_code == 1
        assert "Failed" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_disable(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine

        result = self.runner.invoke(cli, ["watch", "--disable"])
        assert result.exit_code == 0
        assert "disabled" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_default(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.is_auto_watch_enabled.return_value = False

        result = self.runner.invoke(cli, ["watch"])
        assert result.exit_code == 0
        assert "DISABLED" in result.output
        assert "--enable" in result.output


# ---------------------------------------------------------------------------
# cache command
# ---------------------------------------------------------------------------


class TestCacheCmd:
    """Tests for the cache command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["cache", "--help"])
        assert result.exit_code == 0

    @patch("pysearch.cli.main.PySearch")
    def test_stats(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_cache_stats.return_value = {"hit_rate": 0.85, "total_entries": 50}

        result = self.runner.invoke(cli, ["cache", "--stats"])
        assert result.exit_code == 0
        assert "Cache Statistics" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_stats_not_enabled(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.get_cache_stats.return_value = None

        result = self.runner.invoke(cli, ["cache", "--stats"])
        assert result.exit_code == 0
        assert "not enabled" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_clear(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine

        result = self.runner.invoke(cli, ["cache", "--clear"])
        assert result.exit_code == 0
        assert "cleared" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_enable_memory(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.enable_caching.return_value = True

        result = self.runner.invoke(cli, ["cache", "--enable", "memory"])
        assert result.exit_code == 0
        assert "enabled" in result.output
        assert "memory" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_enable_disk(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.enable_caching.return_value = True

        result = self.runner.invoke(
            cli, ["cache", "--enable", "disk", "--cache-dir", "/tmp/cache"]
        )
        assert result.exit_code == 0
        assert "disk" in result.output
        mock_engine.enable_caching.assert_called_once_with(
            backend="disk", cache_dir="/tmp/cache", max_size=1000, default_ttl=3600,
        )

    @patch("pysearch.cli.main.PySearch")
    def test_enable_failure(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.enable_caching.return_value = False

        result = self.runner.invoke(cli, ["cache", "--enable", "memory"])
        assert result.exit_code == 1
        assert "Failed" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_disable(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine

        result = self.runner.invoke(cli, ["cache", "--disable"])
        assert result.exit_code == 0
        assert "disabled" in result.output

    @patch("pysearch.cli.main.PySearch")
    def test_default(self, mock_pysearch):
        mock_engine = MagicMock()
        mock_pysearch.return_value = mock_engine
        mock_engine.is_caching_enabled.return_value = False

        result = self.runner.invoke(cli, ["cache"])
        assert result.exit_code == 0
        assert "DISABLED" in result.output


# ---------------------------------------------------------------------------
# config command
# ---------------------------------------------------------------------------


class TestConfigCmd:
    """Tests for the config command."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0

    @patch("pysearch.cli.main.SearchConfig")
    def test_display_text(self, mock_config_cls):
        mock_cfg = MagicMock()
        mock_cfg.paths = ["."]
        mock_cfg.get_include_patterns.return_value = ["**/*.py"]
        mock_cfg.get_exclude_patterns.return_value = ["**/.git/**"]
        mock_cfg.languages = None
        mock_cfg.context = 2
        mock_cfg.output_format.value = "text"
        mock_cfg.parallel = True
        mock_cfg.workers = 0
        mock_cfg.resolve_cache_dir.return_value = Path(".pysearch-cache")
        mock_cfg.file_size_limit = 2_000_000
        mock_cfg.follow_symlinks = False
        mock_cfg.strict_hash_check = False
        mock_cfg.dir_prune_exclude = True
        mock_cfg.enable_docstrings = True
        mock_cfg.enable_comments = True
        mock_cfg.enable_strings = True
        mock_cfg.is_optional_features_enabled.return_value = False
        mock_config_cls.return_value = mock_cfg

        result = self.runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "Current Configuration" in result.output

    @patch("pysearch.cli.main.SearchConfig")
    def test_display_json(self, mock_config_cls):
        mock_cfg = MagicMock()
        mock_cfg.__dataclass_fields__ = {"paths": None, "context": None}
        mock_cfg.paths = ["."]
        mock_cfg.context = 2
        mock_config_cls.return_value = mock_cfg

        result = self.runner.invoke(cli, ["config", "--format", "json"])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "paths" in data

    @patch("pysearch.cli.main.SearchConfig")
    def test_validate_ok(self, mock_config_cls):
        mock_cfg = MagicMock()
        mock_cfg.validate_optional_config.return_value = []
        mock_config_cls.return_value = mock_cfg

        result = self.runner.invoke(cli, ["config", "--validate"])
        assert result.exit_code == 0
        assert "valid" in result.output

    @patch("pysearch.cli.main.SearchConfig")
    def test_validate_with_issues(self, mock_config_cls):
        mock_cfg = MagicMock()
        mock_cfg.validate_optional_config.return_value = [
            "GraphRAG requires metadata indexing",
        ]
        mock_config_cls.return_value = mock_cfg

        result = self.runner.invoke(cli, ["config", "--validate"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMainFunction:
    """Tests for the main() entry point function."""

    def test_calls_cli(self):
        with patch("pysearch.cli.main.cli") as mock_cli:
            main()
            mock_cli.assert_called_once_with(prog_name="pysearch")
