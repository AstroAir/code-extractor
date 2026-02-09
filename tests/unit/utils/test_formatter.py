"""Tests for pysearch.utils.formatter module."""

from __future__ import annotations

from pathlib import Path

from pysearch.core.types import OutputFormat, SearchItem, SearchResult, SearchStats
from pysearch.utils.formatter import format_result, to_json_bytes


class TestFormatResult:
    """Tests for format_result function."""

    def _make_result(self) -> SearchResult:
        item = SearchItem(
            file=Path("test.py"),
            start_line=1,
            end_line=3,
            lines=["def hello():", "    pass", ""],
        )
        return SearchResult(
            items=[item],
            stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=10.0),
        )

    def test_format_text(self):
        result = self._make_result()
        output = format_result(result, OutputFormat.TEXT)
        assert isinstance(output, str)
        assert "test.py" in output

    def test_format_json(self):
        result = self._make_result()
        output = format_result(result, OutputFormat.JSON)
        assert isinstance(output, str)
        assert "test.py" in output

    def test_format_empty_result(self):
        result = SearchResult(items=[], stats=SearchStats())
        output = format_result(result, OutputFormat.TEXT)
        assert isinstance(output, str)


class TestToJsonBytes:
    """Tests for to_json_bytes function."""

    def test_basic(self):
        item = SearchItem(
            file=Path("test.py"),
            start_line=1,
            end_line=1,
            lines=["x = 1"],
        )
        sr = SearchResult(items=[item], stats=SearchStats(files_scanned=1))
        result = to_json_bytes(sr)
        assert isinstance(result, bytes)
        assert b"test.py" in result

    def test_empty(self):
        sr = SearchResult(items=[], stats=SearchStats())
        result = to_json_bytes(sr)
        assert isinstance(result, bytes)
