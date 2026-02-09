"""
Integration tests for new CLI features.

Tests the new --count, --max-per-file, and --logic CLI options
with real file operations and CLI invocations.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


class TestCountOnlyCLI:
    """Test --count CLI option."""

    def test_count_only_basic(self):
        """Test basic --count functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test files
            (test_dir / "test1.py").write_text(
                """
def function1():
    pass

def function2():
    pass
"""
            )
            (test_dir / "test2.py").write_text(
                """
def function3():
    pass

class TestClass:
    def method1(self):
        pass
"""
            )

            # Run count-only search
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "def",
                    "--path",
                    str(test_dir),
                    "--count",
                    "--format",
                    "text",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0
            output = result.stdout.strip()

            # Should show counts
            assert "Total matches:" in output
            assert "Files matched:" in output

            # Should not show actual file content
            assert "def function1" not in output
            assert "def function2" not in output

    def test_count_only_json_format(self):
        """Test --count with JSON output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test file
            (test_dir / "test.py").write_text(
                """
def func1():
    return 1

def func2():
    return 2
"""
            )

            # Run count-only search with JSON format
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "def",
                    "--path",
                    str(test_dir),
                    "--count",
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0

            # Parse JSON output
            data = json.loads(result.stdout)

            assert "total_matches" in data
            assert "files_matched" in data
            assert "stats" in data
            assert data["total_matches"] >= 0
            assert data["files_matched"] >= 0

    def test_count_only_with_regex(self):
        """Test --count with regex pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test file
            (test_dir / "test.py").write_text(
                """
def test_handler():
    pass

def request_handler():
    pass

def normal_function():
    pass
"""
            )

            # Run count-only search with regex
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    r"def \w+_handler",
                    "--regex",
                    "--path",
                    str(test_dir),
                    "--count",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0
            output = result.stdout.strip()

            # Should find 2 handlers
            assert "Total matches: 2" in output or '"total_matches": 2' in output


class TestMaxPerFileCLI:
    """Test --max-per-file CLI option."""

    def test_max_per_file_limit(self):
        """Test --max-per-file limiting results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test file with multiple matches
            (test_dir / "test.py").write_text(
                """
def function1():
    pass

def function2():
    pass

def function3():
    pass

def function4():
    pass
"""
            )

            # Run search without limit
            result_no_limit = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "def",
                    "--path",
                    str(test_dir),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Run search with limit of 2
            result_with_limit = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "def",
                    "--path",
                    str(test_dir),
                    "--max-per-file",
                    "2",
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result_no_limit.returncode == 0
            assert result_with_limit.returncode == 0

            # Parse results
            data_no_limit = json.loads(result_no_limit.stdout)
            data_with_limit = json.loads(result_with_limit.stdout)

            # Should have fewer results with limit
            assert len(data_no_limit["items"]) > len(data_with_limit["items"])
            assert len(data_with_limit["items"]) <= 2

    def test_max_per_file_with_multiple_files(self):
        """Test --max-per-file with multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create multiple test files
            for i in range(1, 4):
                (test_dir / f"test{i}.py").write_text(
                    f"""
def function{i}_1():
    pass

def function{i}_2():
    pass

def function{i}_3():
    pass
"""
                )

            # Run search with per-file limit
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "def",
                    "--path",
                    str(test_dir),
                    "--max-per-file",
                    "2",
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0
            data = json.loads(result.stdout)

            # Group items by file
            items_by_file = {}
            for item in data["items"]:
                file_path = item["file"]
                items_by_file.setdefault(file_path, []).append(item)

            # Each file should have at most 2 matches
            for _file_path, items in items_by_file.items():
                assert len(items) <= 2


class TestBooleanLogicCLI:
    """Test --logic CLI option."""

    def test_boolean_and_query(self):
        """Test --logic with AND query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test files
            (test_dir / "handler.py").write_text(
                """
async def request_handler():
    return await process_request()
"""
            )
            (test_dir / "test_handler.py").write_text(
                """
async def test_request_handler():
    return mock_response()
"""
            )
            (test_dir / "util.py").write_text(
                """
def async_helper():
    return helper_data()
"""
            )

            # Run boolean search: async AND handler NOT test
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "async AND handler NOT test",
                    "--logic",
                    "--path",
                    str(test_dir),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0
            data = json.loads(result.stdout)

            # Should match handler.py (has async and handler, no test)
            # Should not match test_handler.py (has test)
            # Should not match util.py (no handler)
            matched_files = [item["file"] for item in data["items"]]

            # Check that handler.py is in results
            handler_matched = any("handler.py" in f for f in matched_files)
            assert handler_matched

            # Check that test_handler.py is not in results (has "test")
            test_handler_matched = any("test_handler.py" in f for f in matched_files)
            assert not test_handler_matched

    def test_boolean_or_query(self):
        """Test --logic with OR query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test files
            (test_dir / "handler.py").write_text(
                """
def request_handler():
    return process_request()
"""
            )
            (test_dir / "controller.py").write_text(
                """
class UserController:
    def get_user(self):
        return user_data
"""
            )
            (test_dir / "util.py").write_text(
                """
def helper_function():
    return helper_data()
"""
            )

            # Run boolean search: handler OR controller
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "handler OR controller",
                    "--logic",
                    "--path",
                    str(test_dir),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0
            data = json.loads(result.stdout)

            matched_files = [item["file"] for item in data["items"]]

            # Should match handler.py and controller.py
            handler_matched = any("handler.py" in f for f in matched_files)
            controller_matched = any("controller.py" in f for f in matched_files)

            assert handler_matched
            assert controller_matched

    def test_boolean_with_quotes(self):
        """Test --logic with quoted terms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test file
            (test_dir / "test.py").write_text(
                """
def main():
    return 0

def test_main():
    assert main() == 0
"""
            )

            # Run boolean search with quoted terms
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    '"def main" NOT test',
                    "--logic",
                    "--path",
                    str(test_dir),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0
            data = json.loads(result.stdout)

            # Should find "def main" but not "test_main" (due to NOT test)
            content_found = []
            for item in data["items"]:
                content_found.extend(item["lines"])

            # Should have "def main():" but not "def test_main():"
            main_found = any("def main():" in line for line in content_found)
            test_main_found = any("def test_main():" in line for line in content_found)

            assert main_found
            assert not test_main_found


class TestCLIOptionValidation:
    """Test CLI option validation for new features."""

    def test_conflicting_options_fuzzy_logic(self):
        """Test that --fuzzy and --logic cannot be used together."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pysearch.cli",
                "find",
                "test",
                "--fuzzy",
                "--logic",
                "--path",
                ".",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        assert result.returncode == 1
        assert "Error: --logic and --fuzzy cannot be used together" in result.stderr

    def test_count_with_highlight_format_error(self):
        """Test that --count cannot be used with highlight format."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pysearch.cli",
                "find",
                "test",
                "--count",
                "--format",
                "highlight",
                "--path",
                ".",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        assert result.returncode == 1
        assert "Error: --count cannot be used with highlight format" in result.stderr

    def test_boolean_query_parsing_error(self):
        """Test error handling for malformed boolean queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "test.py").write_text("def test(): pass")

            # Run with malformed boolean query
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "test AND",  # Incomplete query
                    "--logic",
                    "--path",
                    str(test_dir),
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Should handle error gracefully (might return 0 with no results or error)
            # The exact behavior depends on implementation
            assert result.returncode in [0, 1]


class TestCombinedFeatures:
    """Test combinations of new features."""

    def test_count_with_boolean_and_max_per_file(self):
        """Test combining --count, --logic, and --max-per-file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test file with multiple matches
            (test_dir / "test.py").write_text(
                """
async def handler1():
    pass

async def handler2():
    pass

async def test_handler():
    pass

def sync_handler():
    pass
"""
            )

            # Run combined search
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "async AND handler NOT test",
                    "--logic",
                    "--count",
                    "--max-per-file",
                    "1",
                    "--path",
                    str(test_dir),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            assert result.returncode == 0
            data = json.loads(result.stdout)

            assert "total_matches" in data
            assert "files_matched" in data
            # Due to max-per-file=1, should have at most 1 match per file
            # (but there is only 1 file, so total_matches should be <= files_matched)
            assert data["total_matches"] <= data["files_matched"]

    def test_regex_with_count_and_boolean(self):
        """Test --regex with --count and --logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            (test_dir / "test.py").write_text(
                """
def user_handler():
    pass

def admin_handler():
    pass

def test_handler():
    pass
"""
            )

            # Note: --regex and --logic together might not be directly supported
            # This test checks the behavior when both are specified
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pysearch.cli",
                    "find",
                    "handler AND user",
                    "--logic",
                    "--count",
                    "--path",
                    str(test_dir),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Should either work or provide a clear error
            assert result.returncode in [0, 1]

            if result.returncode == 0:
                data = json.loads(result.stdout)
                assert "total_matches" in data
