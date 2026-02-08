"""Tests for pysearch.indexing.metadata.indexer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.indexing.metadata.indexer import MetadataIndexer


class TestMetadataIndexer:
    """Tests for MetadataIndexer class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        assert indexer is not None

    @pytest.mark.asyncio
    async def test_close(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.close()
        # Should not raise
