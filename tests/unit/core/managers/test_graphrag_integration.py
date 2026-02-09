"""Tests for pysearch.core.managers.graphrag_integration module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.managers.graphrag_integration import GraphRAGIntegrationManager


class TestGraphRAGIntegrationManager:
    """Tests for GraphRAGIntegrationManager class."""

    def test_init_defaults(self):
        cfg = SearchConfig()
        mgr = GraphRAGIntegrationManager(cfg)
        assert mgr.config is cfg
        assert mgr.enable_graphrag is False
        assert mgr._graphrag_initialized is False
        assert mgr._graphrag_engine is None
        assert mgr._vector_store is None

    def test_init_enabled(self):
        cfg = SearchConfig(enable_graphrag=True)
        mgr = GraphRAGIntegrationManager(cfg)
        assert mgr.enable_graphrag is True

    def test_init_with_qdrant(self):
        cfg = SearchConfig(enable_graphrag=True)
        qdrant_cfg = {"host": "localhost", "port": 6333}
        mgr = GraphRAGIntegrationManager(cfg, qdrant_config=qdrant_cfg)
        assert mgr.qdrant_config == qdrant_cfg

    def test_set_dependencies(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        logger = MagicMock()
        error_collector = MagicMock()
        mgr.set_dependencies(logger, error_collector)
        assert mgr._logger is logger
        assert mgr._error_collector is error_collector

    def test_is_initialized_default(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        assert mgr.is_initialized() is False

    def test_is_enabled_default(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        assert mgr.is_enabled() is False

    def test_is_enabled_true(self):
        mgr = GraphRAGIntegrationManager(SearchConfig(enable_graphrag=True))
        assert mgr.is_enabled() is True

    def test_get_graph_stats_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        assert mgr.get_graph_stats() == {}

    def test_get_graph_stats_with_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"nodes": 10, "edges": 20}
        mgr._graphrag_engine = mock_engine
        stats = mgr.get_graph_stats()
        assert stats["nodes"] == 10

    def test_get_graph_stats_exception(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        mock_engine = MagicMock()
        mock_engine.get_stats.side_effect = RuntimeError("fail")
        mgr._graphrag_engine = mock_engine
        assert mgr.get_graph_stats() == {}

    def test_get_vector_store_stats_no_store(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        assert mgr.get_vector_store_stats() == {}

    def test_get_vector_store_stats_with_store(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"vectors": 100}
        mgr._vector_store = mock_store
        stats = mgr.get_vector_store_stats()
        assert stats["vectors"] == 100

    @pytest.mark.asyncio
    async def test_initialize_disabled(self):
        mgr = GraphRAGIntegrationManager(SearchConfig(enable_graphrag=False))
        await mgr.initialize()
        assert mgr.is_initialized() is False

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        cfg = SearchConfig(enable_graphrag=True)
        mgr = GraphRAGIntegrationManager(cfg)
        mgr._graphrag_initialized = True
        await mgr.initialize()
        assert mgr._graphrag_engine is None  # should not create new engine

    @pytest.mark.asyncio
    async def test_build_knowledge_graph_disabled(self):
        mgr = GraphRAGIntegrationManager(SearchConfig(enable_graphrag=False))
        result = await mgr.build_knowledge_graph()
        assert result is False

    @pytest.mark.asyncio
    async def test_query_graph_disabled(self):
        mgr = GraphRAGIntegrationManager(SearchConfig(enable_graphrag=False))
        result = await mgr.query_graph("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_close_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        await mgr.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_with_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        mock_engine = AsyncMock()
        mock_store = AsyncMock()
        mgr._graphrag_engine = mock_engine
        mgr._vector_store = mock_store
        await mgr.close()
        mock_engine.close.assert_awaited_once()
        mock_store.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_entities_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.add_entities([])
        assert result is False

    @pytest.mark.asyncio
    async def test_add_entities_with_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        mock_engine = AsyncMock()
        mgr._graphrag_engine = mock_engine
        result = await mgr.add_entities([{"name": "test"}])
        assert result is True
        mock_engine.add_entities.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_relationships_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.add_relationships([])
        assert result is False

    @pytest.mark.asyncio
    async def test_find_similar_entities_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.find_similar_entities("e1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_entity_context_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.get_entity_context("e1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_update_entity_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.update_entity("e1", {"name": "new"})
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_entity_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.delete_entity("e1")
        assert result is False

    @pytest.mark.asyncio
    async def test_export_graph_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.export_graph()
        assert result == ""

    @pytest.mark.asyncio
    async def test_import_graph_no_engine(self):
        mgr = GraphRAGIntegrationManager(SearchConfig())
        result = await mgr.import_graph("{}")
        assert result is False
