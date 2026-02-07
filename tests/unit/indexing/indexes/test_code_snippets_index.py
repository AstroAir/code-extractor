"""Tests for pysearch.indexing.indexes.code_snippets_index module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pysearch.indexing.indexes.code_snippets_index import CodeSnippetsIndex


class TestCodeSnippetsIndex:
    """Tests for CodeSnippetsIndex class."""

    def test_init(self):
        idx = CodeSnippetsIndex(config=MagicMock())
        assert idx is not None

    def test_artifact_id(self):
        idx = CodeSnippetsIndex(config=MagicMock())
        assert isinstance(idx.artifact_id, str)
        assert len(idx.artifact_id) > 0

    def test_relative_expected_time(self):
        idx = CodeSnippetsIndex(config=MagicMock())
        assert idx.relative_expected_time > 0
