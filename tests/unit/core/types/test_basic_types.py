"""Tests for pysearch.core.types.basic_types module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.types.basic_types import (
    ASTFilters,
    BooleanOperator,
    BooleanQuery,
    CountResult,
    FileMetadata,
    MatchSpan,
    MetadataFilters,
    OutputFormat,
    Query,
    SearchItem,
    SearchResult,
    SearchStats,
)
from pysearch.core.types import Language


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_text_value(self):
        assert OutputFormat.TEXT == "text"

    def test_json_value(self):
        assert OutputFormat.JSON == "json"

    def test_highlight_value(self):
        assert OutputFormat.HIGHLIGHT == "highlight"

    def test_is_string_enum(self):
        assert isinstance(OutputFormat.TEXT, str)


class TestASTFilters:
    """Tests for ASTFilters dataclass."""

    def test_defaults(self):
        f = ASTFilters()
        assert f.func_name is None
        assert f.class_name is None
        assert f.decorator is None
        assert f.imported is None

    def test_custom_values(self):
        f = ASTFilters(func_name="main", class_name="App", decorator="cache", imported="os")
        assert f.func_name == "main"
        assert f.class_name == "App"
        assert f.decorator == "cache"
        assert f.imported == "os"

    def test_partial_values(self):
        f = ASTFilters(func_name="test_.*")
        assert f.func_name == "test_.*"
        assert f.class_name is None


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_required_fields(self):
        m = FileMetadata(path=Path("a.py"), size=100, mtime=1.0, language=Language.PYTHON)
        assert m.path == Path("a.py")
        assert m.size == 100
        assert m.mtime == 1.0
        assert m.language == Language.PYTHON

    def test_default_optional_fields(self):
        m = FileMetadata(path=Path("a.py"), size=0, mtime=0.0, language=Language.UNKNOWN)
        assert m.encoding == "utf-8"
        assert m.line_count is None
        assert m.author is None
        assert m.created_date is None
        assert m.modified_date is None

    def test_custom_optional_fields(self):
        m = FileMetadata(
            path=Path("b.js"),
            size=500,
            mtime=2.0,
            language=Language.JAVASCRIPT,
            encoding="ascii",
            line_count=50,
            author="dev",
        )
        assert m.encoding == "ascii"
        assert m.line_count == 50
        assert m.author == "dev"


class TestSearchItem:
    """Tests for SearchItem dataclass."""

    def test_basic_creation(self):
        item = SearchItem(
            file=Path("test.py"),
            start_line=1,
            end_line=3,
            lines=["a", "b", "c"],
        )
        assert item.file == Path("test.py")
        assert item.start_line == 1
        assert item.end_line == 3
        assert len(item.lines) == 3
        assert item.match_spans == []

    def test_with_match_spans(self):
        spans: list[MatchSpan] = [(0, (4, 9)), (1, (0, 5))]
        item = SearchItem(
            file=Path("x.py"),
            start_line=10,
            end_line=11,
            lines=["def hello():", "    pass"],
            match_spans=spans,
        )
        assert len(item.match_spans) == 2
        assert item.match_spans[0] == (0, (4, 9))


class TestSearchStats:
    """Tests for SearchStats dataclass."""

    def test_defaults(self):
        s = SearchStats()
        assert s.files_scanned == 0
        assert s.files_matched == 0
        assert s.items == 0
        assert s.elapsed_ms == 0.0
        assert s.indexed_files == 0

    def test_custom_values(self):
        s = SearchStats(files_scanned=100, files_matched=5, items=10, elapsed_ms=42.5, indexed_files=90)
        assert s.files_scanned == 100
        assert s.elapsed_ms == 42.5


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_defaults(self):
        r = SearchResult()
        assert r.items == []
        assert r.stats.files_scanned == 0

    def test_with_items(self):
        item = SearchItem(file=Path("a.py"), start_line=1, end_line=1, lines=["x"])
        stats = SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0)
        r = SearchResult(items=[item], stats=stats)
        assert len(r.items) == 1
        assert r.stats.items == 1


class TestMetadataFilters:
    """Tests for MetadataFilters dataclass."""

    def test_defaults(self):
        f = MetadataFilters()
        assert f.min_size is None
        assert f.max_size is None
        assert f.modified_after is None
        assert f.modified_before is None
        assert f.min_lines is None
        assert f.max_lines is None
        assert f.author_pattern is None
        assert f.encoding_pattern is None
        assert f.languages is None

    def test_custom_values(self):
        f = MetadataFilters(
            min_size=100,
            max_size=10000,
            min_lines=10,
            languages={Language.PYTHON},
        )
        assert f.min_size == 100
        assert f.max_size == 10000
        assert f.min_lines == 10
        assert Language.PYTHON in f.languages


class TestQuery:
    """Tests for Query dataclass."""

    def test_minimal(self):
        q = Query(pattern="test")
        assert q.pattern == "test"
        assert q.use_regex is False
        assert q.use_ast is False
        assert q.use_semantic is False
        assert q.context == 2
        assert q.output == OutputFormat.TEXT
        assert q.filters is None
        assert q.metadata_filters is None
        assert q.count_only is False
        assert q.max_per_file is None
        assert q.use_boolean is False

    def test_content_toggles_default(self):
        q = Query(pattern="x")
        assert q.search_docstrings is True
        assert q.search_comments is True
        assert q.search_strings is True

    def test_regex_query(self):
        q = Query(pattern=r"def \w+", use_regex=True)
        assert q.use_regex is True

    def test_ast_query(self):
        filters = ASTFilters(func_name="main")
        q = Query(pattern="def", use_ast=True, filters=filters)
        assert q.use_ast is True
        assert q.filters is not None
        assert q.filters.func_name == "main"

    def test_boolean_query(self):
        q = Query(pattern="foo AND bar", use_boolean=True)
        assert q.use_boolean is True

    def test_count_only(self):
        q = Query(pattern="x", count_only=True, max_per_file=5)
        assert q.count_only is True
        assert q.max_per_file == 5


class TestBooleanOperator:
    """Tests for BooleanOperator enum."""

    def test_values(self):
        assert BooleanOperator.AND == "AND"
        assert BooleanOperator.OR == "OR"
        assert BooleanOperator.NOT == "NOT"


class TestBooleanQuery:
    """Tests for BooleanQuery dataclass."""

    def test_leaf_node(self):
        q = BooleanQuery(term="hello")
        assert q.term == "hello"
        assert q.operator is None
        assert q.left is None
        assert q.right is None

    def test_compound_node(self):
        left = BooleanQuery(term="foo")
        right = BooleanQuery(term="bar")
        q = BooleanQuery(operator=BooleanOperator.AND, left=left, right=right)
        assert q.operator == BooleanOperator.AND
        assert q.left.term == "foo"
        assert q.right.term == "bar"


class TestCountResult:
    """Tests for CountResult dataclass."""

    def test_creation(self):
        stats = SearchStats(files_scanned=10, files_matched=3, items=7)
        r = CountResult(total_matches=7, files_matched=3, stats=stats)
        assert r.total_matches == 7
        assert r.files_matched == 3
        assert r.stats.files_scanned == 10
