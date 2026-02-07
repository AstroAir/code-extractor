"""Tests for pysearch.core.integrations.indexing_integration module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.integrations.indexing_integration import IndexingIntegrationManager


class TestIndexingIntegrationManager:
    """Tests for IndexingIntegrationManager class."""

    def test_init_defaults(self):
        cfg = SearchConfig()
        mgr = IndexingIntegrationManager(cfg)
        assert mgr.config is cfg
        assert mgr.enable_metadata_index is False
        assert mgr._indexer is None
        assert mgr._index_initialized is False

    def test_init_enabled(self):
        cfg = SearchConfig(enable_metadata_indexing=True)
        mgr = IndexingIntegrationManager(cfg)
        assert mgr.enable_metadata_index is True

    def test_set_dependencies(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        logger = MagicMock()
        ec = MagicMock()
        mgr.set_dependencies(logger, ec)
        assert mgr._logger is logger
        assert mgr._error_collector is ec

    def test_is_initialized_default(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        assert mgr.is_initialized() is False

    def test_is_enabled_default(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        assert mgr.is_enabled() is False

    def test_is_enabled_true(self):
        mgr = IndexingIntegrationManager(SearchConfig(enable_metadata_indexing=True))
        assert mgr.is_enabled() is True

    def test_get_index_stats_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        assert mgr.get_index_stats() == {}

    def test_get_index_stats_with_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        mock_indexer = MagicMock()
        mock_indexer.get_stats.return_value = {"total_files": 50}
        mgr._indexer = mock_indexer
        stats = mgr.get_index_stats()
        assert stats["total_files"] == 50

    def test_get_index_stats_exception(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        mock_indexer = MagicMock()
        mock_indexer.get_stats.side_effect = RuntimeError("fail")
        mgr._indexer = mock_indexer
        assert mgr.get_index_stats() == {}

    def test_get_index_size_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        assert mgr.get_index_size() == {}

    def test_get_index_size_with_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        mock_indexer = MagicMock()
        mock_indexer.get_size_info.return_value = {"size_bytes": 1024}
        mgr._indexer = mock_indexer
        assert mgr.get_index_size()["size_bytes"] == 1024

    def test_get_index_health_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        result = mgr.get_index_health()
        assert result["status"] == "not_initialized"

    def test_get_index_health_with_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        mock_indexer = MagicMock()
        mock_indexer.get_health_status.return_value = {"status": "healthy"}
        mgr._indexer = mock_indexer
        assert mgr.get_index_health()["status"] == "healthy"

    def test_get_index_health_exception(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        mock_indexer = MagicMock()
        mock_indexer.get_health_status.side_effect = RuntimeError("fail")
        mgr._indexer = mock_indexer
        assert mgr.get_index_health()["status"] == "error"

    @pytest.mark.asyncio
    async def test_initialize_disabled(self):
        mgr = IndexingIntegrationManager(SearchConfig(enable_metadata_indexing=False))
        await mgr.initialize()
        assert mgr.is_initialized() is False

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        cfg = SearchConfig(enable_metadata_indexing=True)
        mgr = IndexingIntegrationManager(cfg)
        mgr._index_initialized = True
        await mgr.initialize()
        assert mgr._indexer is None  # should not create new indexer

    @pytest.mark.asyncio
    async def test_build_index_disabled(self):
        mgr = IndexingIntegrationManager(SearchConfig(enable_metadata_indexing=False))
        result = await mgr.build_index()
        assert result is False

    @pytest.mark.asyncio
    async def test_query_index_disabled(self):
        mgr = IndexingIntegrationManager(SearchConfig(enable_metadata_indexing=False))
        result = await mgr.query_index("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_close_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        await mgr.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_with_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        mock_indexer = AsyncMock()
        mgr._indexer = mock_indexer
        await mgr.close()
        mock_indexer.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_index_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        result = await mgr.update_index(["a.py"])
        assert result is False

    @pytest.mark.asyncio
    async def test_update_index_with_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        mock_indexer = AsyncMock()
        mgr._indexer = mock_indexer
        result = await mgr.update_index(["a.py"])
        assert result is True

    @pytest.mark.asyncio
    async def test_remove_from_index_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        result = await mgr.remove_from_index(["a.py"])
        assert result is False

    @pytest.mark.asyncio
    async def test_optimize_index_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        result = await mgr.optimize_index()
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_index_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        result = await mgr.clear_index()
        assert result is False

    @pytest.mark.asyncio
    async def test_backup_index_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        result = await mgr.backup_index("/tmp/backup")
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_index_no_indexer(self):
        mgr = IndexingIntegrationManager(SearchConfig())
        result = await mgr.restore_index("/tmp/backup")
        assert result is False
