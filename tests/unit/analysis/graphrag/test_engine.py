"""Tests for pysearch.analysis.graphrag.engine module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.analysis.graphrag.engine import (
    GraphRAGEngine,
    KnowledgeGraphBuilder,
)
from pysearch.core.config import SearchConfig


class TestKnowledgeGraphBuilder:
    """Tests for KnowledgeGraphBuilder class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        builder = KnowledgeGraphBuilder(cfg)
        assert builder is not None
        assert builder.config is cfg

    @pytest.mark.asyncio
    async def test_build_graph(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        builder = KnowledgeGraphBuilder(cfg)
        graph = await builder.build_graph()
        assert graph is not None


class TestGraphRAGEngine:
    """Tests for GraphRAGEngine class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        assert engine is not None
        assert engine.config is cfg

    def test_get_stats_empty(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        stats = engine.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_close(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        await engine.close()
