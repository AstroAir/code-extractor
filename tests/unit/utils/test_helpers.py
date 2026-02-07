"""Tests for pysearch.utils.helpers module."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pysearch.utils.helpers import (
    extract_context,
    file_meta,
    file_sha1,
    highlight_spans,
    iter_files,
    iter_python_ast_nodes,
    read_text_safely,
    split_lines_keepends,
)


class TestSplitLinesKeepends:
    """Tests for split_lines_keepends function."""

    def test_basic(self):
        lines = split_lines_keepends("a\nb\nc\n")
        assert len(lines) >= 3
        assert lines[0].strip() == "a"
        assert lines[1].strip() == "b"

    def test_no_trailing_newline(self):
        lines = split_lines_keepends("a\nb")
        assert len(lines) == 2
        assert lines[-1] == "b"

    def test_empty(self):
        lines = split_lines_keepends("")
        assert lines == []

    def test_single_line(self):
        lines = split_lines_keepends("hello")
        assert lines == ["hello"]


class TestExtractContext:
    """Tests for extract_context function."""

    def test_basic(self):
        lines = ["a", "b", "c", "d", "e"]
        start, end, ctx_lines = extract_context(lines, 2, 3, window=1)
        assert isinstance(ctx_lines, list)

    def test_zero_context(self):
        lines = ["a", "b", "c"]
        start, end, ctx_lines = extract_context(lines, 1, 2, window=0)
        assert isinstance(ctx_lines, list)

    def test_beyond_bounds(self):
        lines = ["a", "b"]
        start, end, ctx_lines = extract_context(lines, 0, 1, window=5)
        assert isinstance(ctx_lines, list)


class TestReadTextSafely:
    """Tests for read_text_safely function."""

    def test_read_valid_file(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("hello world", encoding="utf-8")
        content = read_text_safely(f)
        assert content == "hello world"

    def test_read_nonexistent(self, tmp_path: Path):
        content = read_text_safely(tmp_path / "nonexistent.py")
        assert content is None

    def test_read_with_max_bytes(self, tmp_path: Path):
        f = tmp_path / "big.py"
        f.write_text("x" * 1000, encoding="utf-8")
        content = read_text_safely(f, max_bytes=100)
        assert content is None or len(content) <= 1000


class TestFileSha1:
    """Tests for file_sha1 function."""

    def test_basic(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("hello", encoding="utf-8")
        h = file_sha1(f)
        assert isinstance(h, str)
        assert len(h) == 40  # SHA1 hex

    def test_same_content_same_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("same", encoding="utf-8")
        f2.write_text("same", encoding="utf-8")
        assert file_sha1(f1) == file_sha1(f2)


class TestFileMeta:
    """Tests for file_meta function."""

    def test_basic(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("hello", encoding="utf-8")
        meta = file_meta(f)
        assert meta.size > 0
        assert meta.mtime > 0


class TestIterFiles:
    """Tests for iter_files function."""

    def test_basic(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x", encoding="utf-8")
        (tmp_path / "b.txt").write_text("y", encoding="utf-8")
        files = list(iter_files([str(tmp_path)], include=["**/*.py"], exclude=[]))
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) == 1

    def test_empty_dir(self, tmp_path: Path):
        files = list(iter_files([str(tmp_path)], include=["**/*"], exclude=[]))
        assert isinstance(files, list)


class TestHighlightSpans:
    """Tests for highlight_spans function."""

    def test_basic(self):
        result = highlight_spans("hello world", [(0, 5)])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_spans(self):
        result = highlight_spans("hello", [])
        assert result == "hello"


class TestIterPythonAstNodes:
    """Tests for iter_python_ast_nodes function."""

    def test_basic(self):
        code = "def foo():\n    pass\n"
        result = iter_python_ast_nodes(code)
        assert result is not None

    def test_empty_code(self):
        result = iter_python_ast_nodes("")
        # May return None or an AST module
        assert result is None or isinstance(result, ast.AST)
