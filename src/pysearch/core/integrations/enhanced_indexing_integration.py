"""
Enhanced metadata indexing integration.

This module provides integration with enhanced indexing capabilities for
improved search performance and metadata-based queries.

Classes:
    EnhancedIndexingIntegrationManager: Manages enhanced indexing functionality

Key Features:
    - Advanced metadata indexing
    - Fast metadata-based queries
    - Index optimization and maintenance
    - Custom indexing strategies

Example:
    Using enhanced indexing:
        >>> from pysearch.core.integrations.enhanced_indexing_integration import EnhancedIndexingIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig(enable_enhanced_index=True)
        >>> manager = EnhancedIndexingIntegrationManager(config)
        >>> await manager.initialize()
        >>> await manager.build_index()
"""

from __future__ import annotations

from typing import Any

from ..config import SearchConfig


class EnhancedIndexingIntegrationManager:
    """Manages enhanced metadata indexing functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.enable_enhanced_index = config.enable_enhanced_index
        self._enhanced_indexer = None
        self._enhanced_index_initialized = False
        self._logger = None
        self._error_collector = None

    def set_dependencies(self, logger: Any, error_collector: Any) -> None:
        """Set dependencies for logging and error handling."""
        self._logger = logger
        self._error_collector = error_collector

    async def initialize(self) -> None:
        """Initialize enhanced indexing components."""
        if not self.enable_enhanced_index or self._enhanced_index_initialized:
            return

        try:
            if not self._enhanced_indexer:
                from ...indexing.enhanced_indexer import EnhancedIndexer
                self._enhanced_indexer = EnhancedIndexer(self.config)

            await self._enhanced_indexer.initialize()
            self._enhanced_index_initialized = True
            
            if self._logger:
                self._logger.info("Enhanced indexer initialized successfully")

        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to initialize enhanced indexer: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

    async def build_index(self, force_rebuild: bool = False) -> bool:
        """Build the enhanced metadata index."""
        if not self.enable_enhanced_index:
            if self._logger:
                self._logger.warning("Enhanced indexing not enabled")
            return False

        try:
            await self.initialize()
            if self._enhanced_indexer:
                await self._enhanced_indexer.build_index(force_rebuild)
                if self._logger:
                    self._logger.info("Enhanced index built successfully")
                return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to build enhanced index: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

        return False

    async def query_index(self, query: Any) -> Any | None:
        """Perform enhanced index-based search."""
        if not self.enable_enhanced_index:
            if self._logger:
                self._logger.warning("Enhanced indexing not enabled")
            return None

        try:
            await self.initialize()
            if self._enhanced_indexer:
                result = await self._enhanced_indexer.query(query)
                if self._logger:
                    self._logger.debug(f"Enhanced index search completed")
                return result
        except Exception as e:
            if self._logger:
                self._logger.error(f"Enhanced index search failed: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

        return None

    async def close(self) -> None:
        """Close enhanced indexing components properly."""
        if self._enhanced_indexer:
            await self._enhanced_indexer.close()

    def is_initialized(self) -> bool:
        """Check if enhanced indexing is initialized."""
        return self._enhanced_index_initialized

    def is_enabled(self) -> bool:
        """Check if enhanced indexing is enabled."""
        return self.enable_enhanced_index

    def get_index_stats(self) -> dict[str, Any]:
        """Get enhanced index statistics."""
        if not self._enhanced_indexer:
            return {}

        try:
            return self._enhanced_indexer.get_stats()
        except Exception:
            return {}

    async def update_index(self, file_paths: list[str]) -> bool:
        """Update the index for specific files."""
        if not self._enhanced_indexer:
            return False

        try:
            await self._enhanced_indexer.update_files(file_paths)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to update index: {e}")
            return False

    async def remove_from_index(self, file_paths: list[str]) -> bool:
        """Remove files from the index."""
        if not self._enhanced_indexer:
            return False

        try:
            await self._enhanced_indexer.remove_files(file_paths)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to remove from index: {e}")
            return False

    async def optimize_index(self) -> bool:
        """Optimize the enhanced index for better performance."""
        if not self._enhanced_indexer:
            return False

        try:
            await self._enhanced_indexer.optimize()
            if self._logger:
                self._logger.info("Enhanced index optimized successfully")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to optimize index: {e}")
            return False

    async def clear_index(self) -> bool:
        """Clear the entire enhanced index."""
        if not self._enhanced_indexer:
            return False

        try:
            await self._enhanced_indexer.clear()
            if self._logger:
                self._logger.info("Enhanced index cleared successfully")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to clear index: {e}")
            return False

    def get_index_size(self) -> dict[str, Any]:
        """Get information about index size and storage."""
        if not self._enhanced_indexer:
            return {}

        try:
            return self._enhanced_indexer.get_size_info()
        except Exception:
            return {}

    async def backup_index(self, backup_path: str) -> bool:
        """Create a backup of the enhanced index."""
        if not self._enhanced_indexer:
            return False

        try:
            await self._enhanced_indexer.backup(backup_path)
            if self._logger:
                self._logger.info(f"Enhanced index backed up to {backup_path}")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to backup index: {e}")
            return False

    async def restore_index(self, backup_path: str) -> bool:
        """Restore the enhanced index from a backup."""
        if not self._enhanced_indexer:
            return False

        try:
            await self._enhanced_indexer.restore(backup_path)
            if self._logger:
                self._logger.info(f"Enhanced index restored from {backup_path}")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to restore index: {e}")
            return False

    def get_index_health(self) -> dict[str, Any]:
        """Get health status of the enhanced index."""
        if not self._enhanced_indexer:
            return {"status": "not_initialized"}

        try:
            return self._enhanced_indexer.get_health_status()
        except Exception:
            return {"status": "error"}
