"""
Metadata indexing integration.

This module provides integration with metadata indexing capabilities for
improved search performance and metadata-based queries.

Classes:
    IndexingIntegrationManager: Manages metadata indexing functionality

Key Features:
    - Metadata indexing with semantic support
    - Fast metadata-based queries
    - Index optimization and maintenance
    - Custom indexing strategies

Example:
    Using metadata indexing:
        >>> from pysearch.core.integrations.indexing_integration import IndexingIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig(enable_metadata_indexing=True)
        >>> manager = IndexingIntegrationManager(config)
        >>> await manager.initialize()
        >>> await manager.build_index()
"""

from __future__ import annotations

from typing import Any

from ..config import SearchConfig


class IndexingIntegrationManager:
    """Manages metadata indexing functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.enable_metadata_index = config.enable_metadata_indexing
        self._indexer: Any = None
        self._index_initialized = False
        self._logger = None
        self._error_collector = None

    def set_dependencies(self, logger: Any, error_collector: Any) -> None:
        """Set dependencies for logging and error handling."""
        self._logger = logger
        self._error_collector = error_collector

    async def initialize(self) -> None:
        """Initialize metadata indexing components."""
        if not self.enable_metadata_index or self._index_initialized:
            return

        try:
            if not self._indexer:
                from ...indexing.metadata import MetadataIndexer

                self._indexer = MetadataIndexer(self.config)

            if self._indexer:
                await self._indexer.initialize()
            self._index_initialized = True

            if self._logger:
                self._logger.info("Metadata indexer initialized successfully")

        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to initialize metadata indexer: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

    async def build_index(self, force_rebuild: bool = False) -> bool:
        """Build the metadata index."""
        if not self.enable_metadata_index:
            if self._logger:
                self._logger.warning("Metadata indexing not enabled")
            return False

        try:
            await self.initialize()
            if self._indexer:
                await self._indexer.build_index(force_rebuild)
                if self._logger:
                    self._logger.info("Metadata index built successfully")
                return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to build metadata index: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

        return False

    async def query_index(self, query: Any) -> Any | None:
        """Perform index-based search."""
        if not self.enable_metadata_index:
            if self._logger:
                self._logger.warning("Metadata indexing not enabled")
            return None

        try:
            await self.initialize()
            if self._indexer:
                result = await self._indexer.query(query)
                if self._logger:
                    self._logger.debug("Metadata index search completed")
                return result
        except Exception as e:
            if self._logger:
                self._logger.error(f"Metadata index search failed: {e}")
            if self._error_collector:
                self._error_collector.add_error(e)

        return None

    async def close(self) -> None:
        """Close metadata indexing components properly."""
        if self._indexer:
            await self._indexer.close()

    def is_initialized(self) -> bool:
        """Check if metadata indexing is initialized."""
        return self._index_initialized

    def is_enabled(self) -> bool:
        """Check if metadata indexing is enabled."""
        return self.enable_metadata_index

    def get_index_stats(self) -> dict[str, Any]:
        """Get metadata index statistics."""
        if not self._indexer:
            return {}

        try:
            return self._indexer.get_stats()  # type: ignore[no-any-return]
        except Exception:
            return {}

    async def update_index(self, file_paths: list[str]) -> bool:
        """Update the index for specific files."""
        if not self._indexer:
            return False

        try:
            await self._indexer.update_files(file_paths)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to update index: {e}")
            return False

    async def remove_from_index(self, file_paths: list[str]) -> bool:
        """Remove files from the index."""
        if not self._indexer:
            return False

        try:
            await self._indexer.remove_files(file_paths)
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to remove from index: {e}")
            return False

    async def optimize_index(self) -> bool:
        """Optimize the metadata index for better performance."""
        if not self._indexer:
            return False

        try:
            await self._indexer.optimize()
            if self._logger:
                self._logger.info("Metadata index optimized successfully")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to optimize index: {e}")
            return False

    async def clear_index(self) -> bool:
        """Clear the entire metadata index."""
        if not self._indexer:
            return False

        try:
            await self._indexer.clear()
            if self._logger:
                self._logger.info("Metadata index cleared successfully")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to clear index: {e}")
            return False

    def get_index_size(self) -> dict[str, Any]:
        """Get information about index size and storage."""
        if not self._indexer:
            return {}

        try:
            return self._indexer.get_size_info()  # type: ignore[no-any-return]
        except Exception:
            return {}

    async def backup_index(self, backup_path: str) -> bool:
        """Create a backup of the metadata index."""
        if not self._indexer:
            return False

        try:
            await self._indexer.backup(backup_path)
            if self._logger:
                self._logger.info(f"Metadata index backed up to {backup_path}")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to backup index: {e}")
            return False

    async def restore_index(self, backup_path: str) -> bool:
        """Restore the metadata index from a backup."""
        if not self._indexer:
            return False

        try:
            await self._indexer.restore(backup_path)
            if self._logger:
                self._logger.info(f"Metadata index restored from {backup_path}")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to restore index: {e}")
            return False

    def get_index_health(self) -> dict[str, Any]:
        """Get health status of the metadata index."""
        if not self._indexer:
            return {"status": "not_initialized"}

        try:
            return self._indexer.get_health_status()  # type: ignore[no-any-return]
        except Exception:
            return {"status": "error"}
