"""Tests for pysearch.indexing.advanced.engine module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.indexing.advanced.engine import IndexingEngine


class TestIndexingEngine:
    """Tests for IndexingEngine class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = IndexingEngine(cfg)
        assert engine.config is cfg
        assert engine._paused is False

    def test_pause_resume(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = IndexingEngine(cfg)
        engine.pause()
        assert engine._paused is True
        engine.resume()
        assert engine._paused is False

    @pytest.mark.asyncio
    async def test_refresh_index_empty(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = IndexingEngine(cfg)
        updates = []
        async for update in engine.refresh_index():
            updates.append(update)
        # Should complete without error even on empty directory
        assert isinstance(updates, list)
