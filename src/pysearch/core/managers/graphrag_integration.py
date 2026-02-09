"""
GraphRAG knowledge graph integration.

This module provides integration with GraphRAG capabilities for building
and querying knowledge graphs from code analysis.

Classes:
    GraphRAGIntegrationManager: Manages GraphRAG functionality

Key Features:
    - Knowledge graph construction from code
    - Graph-based query and retrieval
    - Entity and relationship extraction
    - Semantic similarity search within graphs

Example:
    Using GraphRAG integration:
        >>> from pysearch.core.managers.graphrag_integration import GraphRAGIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig(enable_graphrag=True)
        >>> manager = GraphRAGIntegrationManager(config)
        >>> await manager.initialize()
        >>> await manager.build_knowledge_graph()
"""

from __future__ import annotations

from typing import Any

from ..config import SearchConfig


class GraphRAGIntegrationManager:
    """Manages GraphRAG knowledge graph functionality."""

    def __init__(self, config: SearchConfig, qdrant_config: Any | None = None) -> None:
        self.config = config
        self.qdrant_config = qdrant_config
        self.enable_graphrag = config.enable_graphrag
        self._graphrag_engine: Any = None
        self._vector_store: Any = None
        self._graphrag_initialized = False
        self._logger: Any = None
        self._error_collector: Any = None

    def set_dependencies(self, logger: Any, error_collector: Any) -> None:
        """Set dependencies for logging and error handling."""
        self._logger = logger
        self._error_collector = error_collector

    async def initialize(self) -> None:
        """Initialize GraphRAG components."""
        if not self.enable_graphrag or self._graphrag_initialized:
            return

        try:
            if not self._graphrag_engine:
                from ...analysis.graphrag.engine import GraphRAGEngine

                self._graphrag_engine = GraphRAGEngine(self.config, self.qdrant_config)

            if self._graphrag_engine:
                await self._graphrag_engine.initialize()
                # Set the vector store reference from the GraphRAG engine
                self._vector_store = self._graphrag_engine.vector_store
            self._graphrag_initialized = True

            if self._logger:
                self._logger.info("GraphRAG engine initialized successfully")

        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to initialize GraphRAG: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

    async def build_knowledge_graph(self, force_rebuild: bool = False) -> bool:
        """Build the GraphRAG knowledge graph."""
        if not self.enable_graphrag:
            if self._logger:
                self._logger.warning("GraphRAG not enabled")
            return False

        try:
            await self.initialize()
            if self._graphrag_engine:
                await self._graphrag_engine.build_knowledge_graph(force_rebuild)
                if self._logger:
                    self._logger.info("Knowledge graph built successfully")
                return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to build knowledge graph: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

        return False

    async def query_graph(self, query: Any) -> Any | None:
        """Perform GraphRAG-based search."""
        if not self.enable_graphrag:
            if self._logger:
                self._logger.warning("GraphRAG not enabled")
            return None

        try:
            await self.initialize()
            if self._graphrag_engine:
                result = await self._graphrag_engine.query_graph(query)
                if self._logger:
                    self._logger.debug(f"GraphRAG search found {len(result.entities)} entities")
                return result
        except Exception as e:
            if self._logger:
                self._logger.error(f"GraphRAG search failed: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

        return None

    async def close(self) -> None:
        """Close GraphRAG components properly."""
        if self._graphrag_engine:
            await self._graphrag_engine.close()

        if self._vector_store:
            await self._vector_store.close()

    def is_initialized(self) -> bool:
        """Check if GraphRAG is initialized."""
        return self._graphrag_initialized

    def is_enabled(self) -> bool:
        """Check if GraphRAG is enabled."""
        return self.enable_graphrag

    def get_graph_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self._graphrag_engine:
            return {}

        try:
            return self._graphrag_engine.get_stats()
        except Exception:
            return {}

    def get_vector_store_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        if not self._vector_store:
            return {}

        try:
            return self._vector_store.get_stats()
        except Exception:
            return {}

    async def add_entities(self, entities: list[Any]) -> bool:
        """Add entities to the knowledge graph."""
        if not self._graphrag_engine:
            return False

        try:
            await self._graphrag_engine.add_entities(entities)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to add entities: {e}")
            return False

    async def add_relationships(self, relationships: list[Any]) -> bool:
        """Add relationships to the knowledge graph."""
        if not self._graphrag_engine:
            return False

        try:
            await self._graphrag_engine.add_relationships(relationships)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to add relationships: {e}")
            return False

    async def find_similar_entities(self, entity_id: str, limit: int = 10) -> list[Any]:
        """Find entities similar to the given entity."""
        if not self._graphrag_engine:
            return []

        try:
            return await self._graphrag_engine.find_similar_entities(entity_id, limit)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to find similar entities: {e}")
            return []

    async def get_entity_context(self, entity_id: str, max_hops: int = 2) -> dict[str, Any]:
        """Get contextual information for an entity."""
        if not self._graphrag_engine:
            return {}

        try:
            return await self._graphrag_engine.get_entity_context(entity_id, max_hops)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to get entity context: {e}")
            return {}

    async def update_entity(self, entity_id: str, updates: dict[str, Any]) -> bool:
        """Update an entity in the knowledge graph."""
        if not self._graphrag_engine:
            return False

        try:
            await self._graphrag_engine.update_entity(entity_id, updates)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to update entity: {e}")
            return False

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from the knowledge graph."""
        if not self._graphrag_engine:
            return False

        try:
            await self._graphrag_engine.delete_entity(entity_id)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to delete entity: {e}")
            return False

    async def export_graph(self, format: str = "json") -> str:
        """Export the knowledge graph in specified format."""
        if not self._graphrag_engine:
            return ""

        try:
            return await self._graphrag_engine.export_graph(format)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to export graph: {e}")
            return ""

    async def import_graph(self, data: str, format: str = "json") -> bool:
        """Import a knowledge graph from data."""
        if not self._graphrag_engine:
            return False

        try:
            await self._graphrag_engine.import_graph(data, format)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to import graph: {e}")
            return False

    async def reset_graph(self) -> bool:
        """Reset the knowledge graph and delete all associated vector data."""
        if not self._graphrag_engine:
            return False

        try:
            await self._graphrag_engine.reset_graph()
            if self._logger:
                self._logger.info("Knowledge graph reset successfully")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to reset graph: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)
            return False

    async def batch_find_similar(
        self, entity_ids: list[str], limit: int = 10
    ) -> dict[str, list[Any]]:
        """Find similar entities for multiple entities using batch vector search.

        Uses Qdrant's batch_search API for efficient multi-query similarity search.

        Args:
            entity_ids: List of entity IDs to find similar entities for.
            limit: Maximum number of similar entities per query.

        Returns:
            Dict mapping entity_id to list of similar entities.
        """
        if not self._graphrag_engine:
            return {eid: [] for eid in entity_ids}

        try:
            return await self._graphrag_engine.batch_find_similar(entity_ids, limit)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to batch find similar entities: {e}")
            return {eid: [] for eid in entity_ids}

    async def get_graph_stats_async(self) -> dict[str, Any]:
        """Get knowledge graph statistics including vector store details (async)."""
        if not self._graphrag_engine:
            return {}

        try:
            return await self._graphrag_engine.get_stats_async()
        except Exception:
            return {}
