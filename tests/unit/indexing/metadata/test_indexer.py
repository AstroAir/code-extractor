"""Tests for pysearch.indexing.metadata.indexer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.indexing.metadata.indexer import MetadataIndexer
from pysearch.indexing.metadata.models import IndexQuery


class TestMetadataIndexer:
    """Tests for MetadataIndexer class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        assert indexer is not None
        assert indexer._initialized is False

    @pytest.mark.asyncio
    async def test_close(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.initialize()
        assert indexer._initialized is True
        await indexer.close()

    @pytest.mark.asyncio
    async def test_query_index_auto_initializes(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        result = await indexer.query_index(IndexQuery())
        assert isinstance(result, dict)
        assert "files" in result
        assert "entities" in result
        assert "stats" in result
        await indexer.close()

    @pytest.mark.asyncio
    async def test_query_alias(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        result = await indexer.query(IndexQuery())
        assert isinstance(result, dict)
        assert "files" in result
        await indexer.close()

    @pytest.mark.asyncio
    async def test_update_files(self, tmp_path: Path):
        f = tmp_path / "hello.py"
        f.write_text("x = 1\n", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.initialize()
        await indexer.update_files([str(f)])
        # Should not raise
        await indexer.close()

    @pytest.mark.asyncio
    async def test_remove_files(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.initialize()
        await indexer.remove_files(["nonexistent.py"])
        # Should not raise
        await indexer.close()

    @pytest.mark.asyncio
    async def test_optimize(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.initialize()
        await indexer.optimize()
        # Should not raise
        await indexer.close()

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.initialize()
        await indexer.clear()
        stats = indexer.get_stats()
        assert stats.get("total_files", 0) == 0
        assert stats.get("total_entities", 0) == 0
        await indexer.close()

    def test_get_stats_not_initialized(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        stats = indexer.get_stats()
        assert isinstance(stats, dict)

    def test_get_size_info(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        info = indexer.get_size_info()
        assert isinstance(info, dict)

    def test_get_health_status_not_initialized(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        status = indexer.get_health_status()
        assert isinstance(status, dict)
        assert status["status"] == "not_initialized"
        assert status["initialized"] is False

    @pytest.mark.asyncio
    async def test_get_health_status_initialized(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.initialize()
        status = indexer.get_health_status()
        assert status["status"] == "healthy"
        assert status["initialized"] is True
        await indexer.close()

    @pytest.mark.asyncio
    async def test_backup_no_db(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        with pytest.raises(FileNotFoundError):
            await indexer.backup(str(tmp_path / "backup.db"))

    @pytest.mark.asyncio
    async def test_backup_and_restore(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        await indexer.initialize()
        backup_path = str(tmp_path / "backup.db")
        await indexer.backup(backup_path)
        assert (tmp_path / "backup.db").exists()
        await indexer.restore(backup_path)
        assert indexer._initialized is True
        await indexer.close()

    @pytest.mark.asyncio
    async def test_restore_missing_file(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = MetadataIndexer(cfg)
        with pytest.raises(FileNotFoundError):
            await indexer.restore(str(tmp_path / "no_such_backup.db"))
