"""Tests for pysearch.indexing.indexes.chunk_index module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.indexing.indexes.chunk_index import ChunkIndex


class TestChunkIndex:
    """Tests for ChunkIndex class."""

    def test_artifact_id(self):
        idx = ChunkIndex(config=MagicMock())
        assert idx.artifact_id == "enhanced_chunks"

    def test_relative_expected_time(self):
        idx = ChunkIndex(config=MagicMock())
        assert idx.relative_expected_time == 1.2

    def test_init(self):
        mock_cfg = MagicMock()
        idx = ChunkIndex(config=mock_cfg)
        assert idx.config is mock_cfg
