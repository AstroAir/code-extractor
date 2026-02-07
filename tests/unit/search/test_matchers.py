"""Tests for pysearch.search.matchers module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.types import ASTFilters, MatchSpan, Query, SearchItem
from pysearch.search.matchers import (
    TextMatch,
    find_ast_blocks,
    find_text_regex_matches,
    group_matches_into_blocks,
    search_in_file,
)


class TestTextMatch:
    """Tests for TextMatch dataclass."""

    def test_creation(self):
        m = TextMatch(line_index=0, start_col=5, end_col=10)
        assert m.line_index == 0
        assert m.start_col == 5
        assert m.end_col == 10


class TestFindTextRegexMatches:
    """Tests for find_text_regex_matches function."""

    def test_simple_text(self):
        matches = find_text_regex_matches("hello world\nhello again", "hello", use_regex=False)
        assert len(matches) >= 2

    def test_no_match(self):
        matches = find_text_regex_matches("hello world", "xyz", use_regex=False)
        assert len(matches) == 0

    def test_regex_match(self):
        matches = find_text_regex_matches("def foo():\n    pass\ndef bar():\n    pass", r"def \w+", use_regex=True)
        assert len(matches) >= 2

    def test_empty_content(self):
        matches = find_text_regex_matches("", "hello", use_regex=False)
        assert matches == []


class TestGroupMatchesIntoBlocks:
    """Tests for group_matches_into_blocks function."""

    def test_adjacent_lines(self):
        matches = [
            TextMatch(line_index=0, start_col=0, end_col=3),
            TextMatch(line_index=1, start_col=0, end_col=3),
        ]
        blocks = group_matches_into_blocks(matches)
        assert len(blocks) == 1

    def test_separate_lines(self):
        matches = [
            TextMatch(line_index=0, start_col=0, end_col=3),
            TextMatch(line_index=10, start_col=0, end_col=3),
        ]
        blocks = group_matches_into_blocks(matches)
        assert len(blocks) == 2

    def test_empty_matches(self):
        blocks = group_matches_into_blocks([])
        assert blocks == []


class TestFindAstBlocks:
    """Tests for find_ast_blocks function."""

    def test_find_function(self):
        content = "def hello():\n    pass\n\ndef world():\n    pass\n"
        blocks = find_ast_blocks(content, ASTFilters(func_name="hello"))
        assert len(blocks) >= 1

    def test_find_class(self):
        content = "class MyClass:\n    def method(self):\n        pass\n"
        blocks = find_ast_blocks(content, ASTFilters(class_name="MyClass"))
        assert len(blocks) >= 1

    def test_no_match(self):
        content = "x = 1\ny = 2\n"
        blocks = find_ast_blocks(content, ASTFilters(func_name="nonexistent"))
        assert isinstance(blocks, list)

    def test_no_filters(self):
        content = "def hello():\n    pass\n"
        blocks = find_ast_blocks(content, None)
        assert isinstance(blocks, list)


class TestSearchInFile:
    """Tests for search_in_file function."""

    def test_text_mode(self):
        content = "hello world"
        query = Query(pattern="hello", use_regex=False, context=0)
        items = search_in_file(Path("test.py"), content, query)
        assert len(items) >= 1
        assert items[0].file == Path("test.py")

    def test_regex_mode(self):
        content = "def foo():\n    pass"
        query = Query(pattern=r"def \w+", use_regex=True, context=0)
        items = search_in_file(Path("test.py"), content, query)
        assert len(items) >= 1

    def test_empty_content(self):
        query = Query(pattern="test", use_regex=False, context=0)
        items = search_in_file(Path("test.py"), "", query)
        assert items == []

    def test_ast_mode(self):
        content = "def hello():\n    pass\n\ndef world():\n    pass\n"
        query = Query(pattern="hello", use_ast=True, context=0)
        items = search_in_file(Path("test.py"), content, query)
        assert isinstance(items, list)
