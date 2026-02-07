"""Tests for pysearch.indexing.indexes.full_text_index module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pysearch.indexing.indexes.full_text_index import FullTextIndex


class TestFullTextIndex:
    """Tests for FullTextIndex class."""

    def test_init(self):
        idx = FullTextIndex(config=MagicMock())
        assert idx is not None

    def test_artifact_id(self):
        idx = FullTextIndex(config=MagicMock())
        assert isinstance(idx.artifact_id, str)
        assert len(idx.artifact_id) > 0

    def test_relative_expected_time(self):
        idx = FullTextIndex(config=MagicMock())
        assert idx.relative_expected_time > 0
