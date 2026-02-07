"""Tests for pysearch.analysis.content_addressing module."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pysearch.analysis.content_addressing import (
    ContentAddress,
    ContentAddressedIndexer,
    GlobalCacheManager,
    IndexTag,
    IndexingProgressUpdate,
    MarkCompleteCallback,
    PathAndCacheKey,
    RefreshIndexResults,
)
from pysearch.core.config import SearchConfig


class TestContentAddress:
    """Tests for ContentAddress dataclass."""

    def test_creation(self):
        ca = ContentAddress(
            path="test.py",
            content_hash="abc123",
            size=100,
            mtime=1.0,
        )
        assert ca.path == "test.py"
        assert ca.content_hash == "abc123"
        assert ca.size == 100


class TestIndexTag:
    """Tests for IndexTag dataclass."""

    def test_creation(self):
        tag = IndexTag(
            directory="/repo",
            branch="main",
            artifact_id="chunks",
        )
        assert tag.directory == "/repo"
        assert tag.branch == "main"
        assert tag.artifact_id == "chunks"

    def test_equality(self):
        t1 = IndexTag(directory="/a", branch="main", artifact_id="x")
        t2 = IndexTag(directory="/a", branch="main", artifact_id="x")
        assert t1 == t2

    def test_inequality(self):
        t1 = IndexTag(directory="/a", branch="main", artifact_id="x")
        t2 = IndexTag(directory="/a", branch="dev", artifact_id="x")
        assert t1 != t2


class TestGlobalCacheManager:
    """Tests for GlobalCacheManager class."""

    def test_init(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        assert mgr.cache_dir == tmp_path

    @pytest.mark.asyncio
    async def test_store_and_get(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        await mgr.store_cached_content("hash123", "chunks", {"key": "value"}, [tag])
        result = await mgr.get_cached_content("hash123", "chunks")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        result = await mgr.get_cached_content("nonexistent", "chunks")
        assert result is None


class TestIndexingProgressUpdate:
    """Tests for IndexingProgressUpdate dataclass."""

    def test_creation(self):
        update = IndexingProgressUpdate(
            progress=0.5,
            description="Processing files",
            status="indexing",
        )
        assert update.progress == 0.5
        assert update.description == "Processing files"
        assert update.status == "indexing"


class TestRefreshIndexResults:
    """Tests for RefreshIndexResults dataclass."""

    def test_creation(self):
        results = RefreshIndexResults(
            compute=[],
            delete=[],
            add_tag=[],
            remove_tag=[],
        )
        assert results.compute == []
        assert results.delete == []
