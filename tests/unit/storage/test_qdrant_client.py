"""Tests for pysearch.storage.qdrant_client module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.storage.qdrant_client import (
    QdrantConfig,
    QdrantVectorStore,
)


class TestQdrantConfig:
    """Tests for QdrantConfig dataclass."""

    def test_defaults(self):
        cfg = QdrantConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6333

    def test_custom(self):
        cfg = QdrantConfig(host="remote.server", port=6334)
        assert cfg.host == "remote.server"
        assert cfg.port == 6334


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore class."""

    def test_init(self):
        cfg = QdrantConfig()
        store = QdrantVectorStore(cfg)
        assert store is not None
        assert store.config is cfg

    def test_is_available(self):
        cfg = QdrantConfig()
        store = QdrantVectorStore(cfg)
        assert isinstance(store.is_available(), bool)

    @pytest.mark.asyncio
    async def test_close_not_initialized(self):
        cfg = QdrantConfig()
        store = QdrantVectorStore(cfg)
        await store.close()  # should not raise
