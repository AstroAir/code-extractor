"""Tests for pysearch.indexing.metadata.database module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.indexing.metadata.database import MetadataIndex
from pysearch.indexing.metadata.models import IndexQuery, IndexStats


class TestMetadataIndex:
    """Tests for MetadataIndex class."""

    def test_init(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        assert idx is not None
        assert idx.db_path == tmp_path / "metadata.db"

    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        assert idx._connection is not None

    @pytest.mark.asyncio
    async def test_close(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.close()
        # Should not raise
