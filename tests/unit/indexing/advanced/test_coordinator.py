"""Tests for pysearch.indexing.advanced.coordinator module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.indexing.advanced.coordinator import IndexCoordinator


class TestIndexCoordinator:
    """Tests for IndexCoordinator class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        coord = IndexCoordinator(cfg)
        assert coord.config is cfg
        assert coord.indexes == []

    def test_add_index(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        coord = IndexCoordinator(cfg)
        mock_index = MagicMock()
        mock_index.artifact_id = "test_idx"
        coord.add_index(mock_index)
        assert len(coord.indexes) == 1

    def test_add_multiple_indexes(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        coord = IndexCoordinator(cfg)
        for i in range(3):
            mock_idx = MagicMock()
            mock_idx.artifact_id = f"idx_{i}"
            coord.add_index(mock_idx)
        assert len(coord.indexes) == 3
