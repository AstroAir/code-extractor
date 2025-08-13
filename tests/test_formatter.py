"""
Tests for formatter module.

This module tests the output formatting functionality including
JSON, text, and highlighted output formats.
"""

import json
from pathlib import Path
from unittest.mock import patch

from pysearch.formatter import (
    format_result,
    format_text,
    render_highlight_console,
    to_json_bytes,
)
from pysearch.types import OutputFormat, SearchItem, SearchResult, SearchStats


class TestFormatter:
    """Test output formatting functionality."""

    def create_sample_result(self) -> SearchResult:
        """Create a sample search result for testing."""
        items = [
            SearchItem(
                file=Path("test1.py"),
                start_line=1,
                end_line=3,
                lines=["def hello():", "    print('Hello')", "    return True"],
                match_spans=[(0, (4, 9)), (1, (11, 16))],  # "hello" and "Hello"
            ),
            SearchItem(
                file=Path("test2.py"),
                start_line=5,
                end_line=6,
                lines=["class HelloWorld:", "    pass"],
                match_spans=[(0, (6, 11))],  # "Hello"
            ),
        ]
        stats = SearchStats(
            files_scanned=10, files_matched=2, items=2, elapsed_ms=15.5, indexed_files=8
        )
        return SearchResult(items=items, stats=stats)

    def test_to_json_bytes(self):
        """Test JSON bytes conversion."""
        result = self.create_sample_result()
        json_bytes = to_json_bytes(result)

        # Should return bytes
        assert isinstance(json_bytes, bytes)

        # Should be valid JSON
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)

        # Verify structure
        assert "items" in data
        assert "stats" in data
        assert len(data["items"]) == 2

        # Verify item structure
        item = data["items"][0]
        assert "file" in item
        assert "start_line" in item
        assert "end_line" in item
        assert "lines" in item
        assert "match_spans" in item

        # Verify content
        assert item["file"] == "test1.py"
        assert item["start_line"] == 1
        assert item["end_line"] == 3
        assert len(item["lines"]) == 3
        assert len(item["match_spans"]) == 2

        # Verify stats
        stats = data["stats"]
        assert stats["files_scanned"] == 10
        assert stats["files_matched"] == 2
        assert stats["items"] == 2
        assert stats["elapsed_ms"] == 15.5
        assert stats["indexed_files"] == 8

    def test_format_text_basic(self):
        """Test basic text formatting."""
        result = self.create_sample_result()
        text = format_text(result, highlight=False)

        # Should contain file headers
        assert "test1.py:1-3" in text
        assert "test2.py:5-6" in text

        # Should contain line numbers
        assert "     1 |" in text
        assert "     2 |" in text
        assert "     3 |" in text
        assert "     5 |" in text
        assert "     6 |" in text

        # Should contain code content
        assert "def hello():" in text
        assert "print('Hello')" in text
        assert "class HelloWorld:" in text

        # Should contain stats
        assert "files_scanned=10" in text
        assert "files_matched=2" in text
        assert "items=2" in text
        assert "elapsed_ms=15.50" in text
        assert "indexed=8" in text

    def test_format_text_with_highlight(self):
        """Test text formatting with highlighting."""
        result = self.create_sample_result()
        text = format_text(result, highlight=True)

        # Should contain highlight markers
        assert "[[" in text
        assert "]]" in text

        # Should highlight the matched spans
        # The "hello" in "def hello():" should be highlighted
        assert "def [[hello]]():" in text

        # The "Hello" in "print('Hello')" should be highlighted
        assert "print('[[Hello]]')" in text

    def test_format_text_empty_result(self):
        """Test text formatting with empty result."""
        empty_result = SearchResult(
            items=[],
            stats=SearchStats(
                files_scanned=5, files_matched=0, items=0, elapsed_ms=2.1, indexed_files=5
            ),
        )

        text = format_text(empty_result, highlight=False)

        # Should only contain stats
        assert "files_scanned=5" in text
        assert "files_matched=0" in text
        assert "items=0" in text
        assert "elapsed_ms=2.10" in text
        assert "indexed=5" in text

        # Should not contain any file content
        assert ".py:" not in text

    def test_format_result_json(self):
        """Test format_result with JSON output."""
        result = self.create_sample_result()
        formatted = format_result(result, OutputFormat.JSON)

        # Should be valid JSON string
        data = json.loads(formatted)
        assert "items" in data
        assert "stats" in data
        assert len(data["items"]) == 2

    def test_format_result_text(self):
        """Test format_result with text output."""
        result = self.create_sample_result()
        formatted = format_result(result, OutputFormat.TEXT)

        # Should be plain text without highlights
        assert "[[" not in formatted
        assert "]]" not in formatted
        assert "test1.py:1-3" in formatted

    def test_format_result_highlight(self):
        """Test format_result with highlight output."""
        result = self.create_sample_result()
        formatted = format_result(result, OutputFormat.HIGHLIGHT)

        # Should contain highlight markers
        assert "[[" in formatted
        assert "]]" in formatted
        assert "test1.py:1-3" in formatted

    @patch("pysearch.formatter.Console")
    def test_render_highlight_console(self, mock_console_class):
        """Test console highlighting rendering."""
        mock_console = mock_console_class.return_value
        result = self.create_sample_result()

        render_highlight_console(result)

        # Should create console instance
        mock_console_class.assert_called_once()

        # Should call print methods
        assert mock_console.print.call_count >= 3  # At least headers + stats

    def test_to_json_bytes_with_path_objects(self):
        """Test JSON conversion handles Path objects correctly."""
        # Create result with Path objects
        items = [
            SearchItem(
                file=Path("/absolute/path/test.py"),
                start_line=1,
                end_line=1,
                lines=["test line"],
                match_spans=[],
            )
        ]
        stats = SearchStats(
            files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0, indexed_files=1
        )
        result = SearchResult(items=items, stats=stats)

        json_bytes = to_json_bytes(result)
        data = json.loads(json_bytes.decode("utf-8"))

        # Path should be converted to string (platform-agnostic check)
        file_path = data["items"][0]["file"]
        assert isinstance(file_path, str)
        assert "absolute" in file_path and "path" in file_path and "test.py" in file_path

    def test_format_text_complex_spans(self):
        """Test text formatting with complex match spans."""
        items = [
            SearchItem(
                file=Path("complex.py"),
                start_line=1,
                end_line=2,
                lines=["def process_data_handler():", "    return process_result()"],
                match_spans=[
                    (0, (4, 11)),  # "process" in line 1
                    (0, (17, 24)),  # "handler" in line 1
                    (1, (11, 18)),  # "process" in line 2
                ],
            )
        ]
        stats = SearchStats(
            files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0, indexed_files=1
        )
        result = SearchResult(items=items, stats=stats)

        text = format_text(result, highlight=True)

        # Should highlight multiple spans on same line
        assert "def [[process]]_data_[[handler]]():" in text
        assert "return [[process]]_result()" in text

    def test_json_match_spans_format(self):
        """Test that JSON output formats match spans correctly."""
        result = self.create_sample_result()
        json_bytes = to_json_bytes(result)
        data = json.loads(json_bytes.decode("utf-8"))

        # Verify match_spans format in JSON
        item = data["items"][0]
        spans = item["match_spans"]

        # Should be list of [line_idx, [start, end]]
        assert isinstance(spans, list)
        assert len(spans) == 2

        span1 = spans[0]
        assert len(span1) == 2
        assert span1[0] == 0  # line index
        assert span1[1] == [4, 9]  # [start, end]

        span2 = spans[1]
        assert span2[0] == 1
        assert span2[1] == [11, 16]
