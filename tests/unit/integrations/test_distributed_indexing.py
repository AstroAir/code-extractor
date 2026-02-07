"""Tests for pysearch.integrations.distributed_indexing module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.integrations.distributed_indexing import (
    WorkItem,
    WorkItemType,
    WorkQueue,
)


class TestWorkItemType:
    """Tests for WorkItemType enum."""

    def test_values(self):
        assert WorkItemType.FILE_DISCOVERY == "file_discovery"
        assert WorkItemType.CONTENT_EXTRACTION == "content_extraction"
        assert WorkItemType.ENTITY_PARSING == "entity_parsing"
        assert WorkItemType.CHUNKING == "chunking"
        assert WorkItemType.EMBEDDING_GENERATION == "embedding_generation"
        assert WorkItemType.INDEX_UPDATE == "index_update"


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


class TestWorkQueue:
    """Tests for WorkQueue class."""

    def test_init(self):
        q = WorkQueue()
        assert q is not None

    @pytest.mark.asyncio
    async def test_add_and_get(self):
        q = WorkQueue()
        item = WorkItem(item_id="w1", item_type=WorkItemType.FILE_DISCOVERY)
        await q.add_work_item(item)
        result = await q.get_work_item(timeout=1.0)
        assert result is not None
        assert result.item_id == "w1"

    @pytest.mark.asyncio
    async def test_complete_work_item(self):
        q = WorkQueue()
        item = WorkItem(item_id="w1", item_type=WorkItemType.CHUNKING)
        await q.add_work_item(item)
        got = await q.get_work_item(timeout=1.0)
        assert got is not None
        q.complete_work_item(got.item_id)
        stats = q.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats(self):
        q = WorkQueue()
        stats = q.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_get_empty(self):
        q = WorkQueue()
        result = await q.get_work_item(timeout=0.01)
        assert result is None
