"""Tests for pysearch.core.managers.distributed_indexing_integration module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.managers.distributed_indexing_integration import DistributedIndexingManager


class TestDistributedIndexingManager:
    """Tests for DistributedIndexingManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = DistributedIndexingManager(cfg)
        assert mgr.config is cfg
        assert mgr._engine is None
        assert mgr._enabled is False

    def test_is_distributed_enabled_default(self):
        mgr = DistributedIndexingManager(SearchConfig())
        assert mgr.is_distributed_enabled() is False

    def test_enable_distributed_indexing_import_failure(self):
        mgr = DistributedIndexingManager(SearchConfig())
        result = mgr.enable_distributed_indexing(num_workers=2)
        # The integrations.distributed_indexing module may not be available
        assert isinstance(result, bool)

    def test_enable_distributed_indexing_already_enabled(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._enabled = True
        result = mgr.enable_distributed_indexing()
        assert result is True

    def test_disable_distributed_indexing_when_not_enabled(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr.disable_distributed_indexing()
        assert mgr._enabled is False
        assert mgr._engine is None

    def test_disable_distributed_indexing_clears_state(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._enabled = True
        mgr._engine = MagicMock()
        mgr._engine.stop_workers = AsyncMock()
        mgr.disable_distributed_indexing()
        assert mgr._enabled is False
        assert mgr._engine is None

    # --- Tests for disabled operations returning empty ---

    @pytest.mark.asyncio
    async def test_index_codebase_disabled(self):
        mgr = DistributedIndexingManager(SearchConfig())
        result = await mgr.index_codebase(directories=["/tmp"])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_worker_stats_disabled(self):
        mgr = DistributedIndexingManager(SearchConfig())
        result = await mgr.get_worker_stats()
        assert result == []

    @pytest.mark.asyncio
    async def test_scale_workers_disabled(self):
        mgr = DistributedIndexingManager(SearchConfig())
        result = await mgr.scale_workers(4)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_performance_metrics_disabled(self):
        mgr = DistributedIndexingManager(SearchConfig())
        result = await mgr.get_performance_metrics()
        assert result == {}

    def test_get_queue_stats_disabled(self):
        mgr = DistributedIndexingManager(SearchConfig())
        result = mgr.get_queue_stats()
        assert result == {}

    # --- Tests with mocked engine ---

    @pytest.mark.asyncio
    async def test_get_worker_stats_exception(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.get_worker_stats = AsyncMock(side_effect=RuntimeError("fail"))
        result = await mgr.get_worker_stats()
        assert result == []

    @pytest.mark.asyncio
    async def test_scale_workers_exception(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.scale_workers = AsyncMock(side_effect=RuntimeError("fail"))
        result = await mgr.scale_workers(4)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_performance_metrics_exception(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.get_performance_metrics = AsyncMock(side_effect=RuntimeError("fail"))
        result = await mgr.get_performance_metrics()
        assert result == {}

    def test_get_queue_stats_exception(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.work_queue.get_stats.side_effect = RuntimeError("fail")
        result = mgr.get_queue_stats()
        assert result == {}

    @pytest.mark.asyncio
    async def test_scale_workers_success(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.scale_workers = AsyncMock()
        result = await mgr.scale_workers(4)
        assert result is True
        mgr._engine.scale_workers.assert_called_once_with(4)

    @pytest.mark.asyncio
    async def test_get_performance_metrics_success(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.get_performance_metrics = AsyncMock(
            return_value={"workers": 4, "throughput": 100.0}
        )
        result = await mgr.get_performance_metrics()
        assert result == {"workers": 4, "throughput": 100.0}

    def test_get_queue_stats_success(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.work_queue.get_stats.return_value = {"pending": 10, "completed": 50}
        result = mgr.get_queue_stats()
        assert result == {"pending": 10, "completed": 50}

    @pytest.mark.asyncio
    async def test_index_codebase_exception(self):
        mgr = DistributedIndexingManager(SearchConfig())
        mgr._engine = MagicMock()
        mgr._engine.index_codebase = MagicMock(side_effect=RuntimeError("fail"))
        result = await mgr.index_codebase(directories=["/tmp"])
        assert result == []
