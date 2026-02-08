"""Tests for pysearch.integrations.distributed_indexing module."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.core.config import SearchConfig
from pysearch.integrations.distributed_indexing import (
    DistributedIndexingEngine,
    IndexingWorker,
    WorkItem,
    WorkItemType,
    WorkQueue,
    WorkerStats,
)


# ---------------------------------------------------------------------------
# WorkItemType
# ---------------------------------------------------------------------------

class TestWorkItemType:
    """Tests for WorkItemType enum."""

    def test_values(self):
        assert WorkItemType.FILE_DISCOVERY == "file_discovery"
        assert WorkItemType.CONTENT_EXTRACTION == "content_extraction"
        assert WorkItemType.ENTITY_PARSING == "entity_parsing"
        assert WorkItemType.CHUNKING == "chunking"
        assert WorkItemType.EMBEDDING_GENERATION == "embedding_generation"
        assert WorkItemType.INDEX_UPDATE == "index_update"

    def test_is_string_enum(self):
        assert isinstance(WorkItemType.FILE_DISCOVERY, str)

    def test_all_members(self):
        assert len(WorkItemType) == 6


# ---------------------------------------------------------------------------
# WorkItem
# ---------------------------------------------------------------------------

class TestWorkItem:
    """Tests for WorkItem dataclass."""

    def test_creation(self):
        item = WorkItem(
            item_id="w1",
            item_type=WorkItemType.FILE_DISCOVERY,
        )
        assert item.item_id == "w1"
        assert item.item_type == WorkItemType.FILE_DISCOVERY
        assert item.priority == 0

    def test_defaults(self):
        item = WorkItem(item_id="w1", item_type=WorkItemType.CHUNKING)
        assert item.data == {}
        assert item.dependencies == []
        assert item.started_at is None
        assert item.completed_at is None
        assert item.worker_id is None
        assert item.error is None

    def test_with_priority(self):
        item = WorkItem(
            item_id="w1",
            item_type=WorkItemType.EMBEDDING_GENERATION,
            priority=10,
        )
        assert item.priority == 10

    def test_with_data_and_dependencies(self):
        item = WorkItem(
            item_id="w2",
            item_type=WorkItemType.CONTENT_EXTRACTION,
            data={"file_path": "/tmp/test.py"},
            dependencies=["w1"],
        )
        assert item.data["file_path"] == "/tmp/test.py"
        assert "w1" in item.dependencies

    def test_created_at_auto_set(self):
        before = time.time()
        item = WorkItem(item_id="w1", item_type=WorkItemType.CHUNKING)
        after = time.time()
        assert before <= item.created_at <= after


# ---------------------------------------------------------------------------
# WorkerStats
# ---------------------------------------------------------------------------

class TestWorkerStats:
    """Tests for WorkerStats dataclass."""

    def test_creation(self):
        stats = WorkerStats(worker_id="w0", process_id=1234)
        assert stats.worker_id == "w0"
        assert stats.process_id == 1234

    def test_defaults(self):
        stats = WorkerStats(worker_id="w0", process_id=1)
        assert stats.items_processed == 0
        assert stats.items_failed == 0
        assert stats.total_processing_time == 0.0
        assert stats.current_item is None
        assert stats.last_heartbeat > 0
        assert stats.memory_usage_mb == 0.0
        assert stats.cpu_usage_percent == 0.0


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------

class TestWorkQueue:
    """Tests for WorkQueue class."""

    def test_init(self):
        q = WorkQueue()
        assert q.max_size == 10000

    def test_init_custom_size(self):
        q = WorkQueue(max_size=50)
        assert q.max_size == 50

    @pytest.mark.asyncio
    async def test_add_and_get(self):
        q = WorkQueue()
        item = WorkItem(item_id="w1", item_type=WorkItemType.FILE_DISCOVERY)
        added = await q.add_work_item(item)
        assert added is True
        result = await q.get_work_item(timeout=1.0)
        assert result is not None
        assert result.item_id == "w1"
        assert result.started_at is not None

    @pytest.mark.asyncio
    async def test_complete_work_item(self):
        q = WorkQueue()
        item = WorkItem(item_id="w1", item_type=WorkItemType.CHUNKING)
        await q.add_work_item(item)
        got = await q.get_work_item(timeout=1.0)
        assert got is not None
        await q.complete_work_item(got.item_id)
        stats = q.get_stats()
        assert stats["completed_items"] == 1
        assert stats["pending_items"] == 0

    @pytest.mark.asyncio
    async def test_fail_work_item(self):
        q = WorkQueue()
        item = WorkItem(item_id="w1", item_type=WorkItemType.ENTITY_PARSING)
        await q.add_work_item(item)
        await q.get_work_item(timeout=1.0)
        await q.fail_work_item("w1", "some error")
        stats = q.get_stats()
        assert stats["failed_items"] == 1

    @pytest.mark.asyncio
    async def test_add_with_unmet_dependency(self):
        q = WorkQueue()
        item = WorkItem(
            item_id="w2",
            item_type=WorkItemType.CONTENT_EXTRACTION,
            dependencies=["w1"],
        )
        added = await q.add_work_item(item)
        assert added is False

    @pytest.mark.asyncio
    async def test_add_with_met_dependency(self):
        q = WorkQueue()
        dep = WorkItem(item_id="w1", item_type=WorkItemType.FILE_DISCOVERY)
        await q.add_work_item(dep)
        await q.get_work_item()
        await q.complete_work_item("w1")

        child = WorkItem(
            item_id="w2",
            item_type=WorkItemType.CONTENT_EXTRACTION,
            dependencies=["w1"],
        )
        added = await q.add_work_item(child)
        assert added is True

    def test_get_stats_initial(self):
        q = WorkQueue()
        stats = q.get_stats()
        assert stats == {
            "queue_size": 0,
            "pending_items": 0,
            "completed_items": 0,
            "failed_items": 0,
        }

    @pytest.mark.asyncio
    async def test_get_empty_queue(self):
        q = WorkQueue()
        result = await q.get_work_item(timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        q = WorkQueue()
        low = WorkItem(item_id="low", item_type=WorkItemType.CHUNKING, priority=1)
        high = WorkItem(item_id="high", item_type=WorkItemType.CHUNKING, priority=10)
        await q.add_work_item(low)
        await q.add_work_item(high)

        first = await q.get_work_item()
        assert first is not None
        assert first.item_id == "high"

        second = await q.get_work_item()
        assert second is not None
        assert second.item_id == "low"


# ---------------------------------------------------------------------------
# IndexingWorker
# ---------------------------------------------------------------------------

class TestIndexingWorker:
    """Tests for IndexingWorker class."""

    def test_init(self):
        cfg = SearchConfig()
        worker = IndexingWorker(
            worker_id="w0",
            config=cfg,
            work_types=[WorkItemType.FILE_DISCOVERY],
        )
        assert worker.worker_id == "w0"
        assert worker.running is False
        assert WorkItemType.FILE_DISCOVERY in worker.work_types

    def test_stop(self):
        cfg = SearchConfig()
        worker = IndexingWorker(
            worker_id="w0",
            config=cfg,
            work_types=[WorkItemType.CHUNKING],
        )
        worker.running = True
        worker.stop()
        assert worker.running is False

    def test_stats_initialized(self):
        cfg = SearchConfig()
        worker = IndexingWorker(
            worker_id="w1",
            config=cfg,
            work_types=list(WorkItemType),
        )
        assert worker.stats.worker_id == "w1"
        assert worker.stats.items_processed == 0
        assert worker.stats.items_failed == 0


# ---------------------------------------------------------------------------
# DistributedIndexingEngine
# ---------------------------------------------------------------------------

class TestDistributedIndexingEngine:
    """Tests for DistributedIndexingEngine class."""

    def test_init_defaults(self):
        cfg = SearchConfig()
        engine = DistributedIndexingEngine(cfg)
        assert engine.num_workers == min(mp.cpu_count(), 8)
        assert engine.running is False
        assert engine.workers == []
        assert engine.worker_tasks == []

    def test_init_custom_workers(self):
        cfg = SearchConfig()
        engine = DistributedIndexingEngine(cfg, num_workers=2, max_queue_size=500)
        assert engine.num_workers == 2
        assert engine.work_queue.max_size == 500

    @pytest.mark.asyncio
    async def test_get_worker_stats_empty(self):
        cfg = SearchConfig()
        engine = DistributedIndexingEngine(cfg, num_workers=1)
        stats = await engine.get_worker_stats()
        assert stats == []

    @pytest.mark.asyncio
    async def test_get_performance_metrics_no_workers(self):
        cfg = SearchConfig()
        engine = DistributedIndexingEngine(cfg, num_workers=1)
        metrics = await engine.get_performance_metrics()
        assert metrics["workers"]["total_workers"] == 0
        assert metrics["workers"]["average_processing_time"] == 0.0
        assert "queue" in metrics
        assert "performance" in metrics

    @pytest.mark.asyncio
    async def test_stop_workers_when_not_running(self):
        cfg = SearchConfig()
        engine = DistributedIndexingEngine(cfg, num_workers=1)
        await engine.stop_workers()
        assert engine.running is False
