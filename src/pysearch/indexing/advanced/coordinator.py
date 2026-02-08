"""
Index coordination for the enhanced indexing system.

This module provides the IndexCoordinator class that manages multiple
index types, ensuring they work together efficiently and handle updates
correctly.

Classes:
    IndexCoordinator: Coordinates multiple index types

Features:
    - Multiple index type management
    - Coordinated updates across all indexes
    - Progress tracking and reporting
    - Error handling and recovery
    - Lock management for concurrent operations
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from ...analysis.content_addressing import (
    ContentAddress,
    GlobalCacheManager,
    IndexingProgressUpdate,
    IndexTag,
    PathAndCacheKey,
)
from ...core.config import SearchConfig
from ...utils.error_handling import CircuitBreaker, ErrorCollector
from ...utils.logging_config import get_logger
from .base import CodebaseIndex
from .locking import IndexLock

logger = get_logger()


class IndexCoordinator:
    """
    Coordinates multiple index types for comprehensive code indexing.

    This class manages the lifecycle of multiple index implementations,
    ensuring they work together efficiently and handle updates correctly.
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.indexes: list[CodebaseIndex] = []
        self.global_cache = GlobalCacheManager(config.resolve_cache_dir())
        self.error_collector = ErrorCollector()
        self.lock = IndexLock(config.resolve_cache_dir())
        # Circuit breakers per index to prevent cascading failures
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def add_index(self, index: CodebaseIndex) -> None:
        """Add an index to the coordinator."""
        self.indexes.append(index)
        logger.info(f"Added index: {index.artifact_id}")

    def remove_index(self, artifact_id: str) -> bool:
        """Remove an index by artifact ID."""
        for i, index in enumerate(self.indexes):
            if index.artifact_id == artifact_id:
                del self.indexes[i]
                logger.info(f"Removed index: {artifact_id}")
                return True
        return False

    def get_index(self, artifact_id: str) -> CodebaseIndex | None:
        """Get an index by artifact ID."""
        for index in self.indexes:
            if index.artifact_id == artifact_id:
                return index
        return None

    async def refresh_all_indexes(
        self,
        tag: IndexTag,
        current_files: dict[str, Any],
        read_file: Callable[[str], str],
        repo_name: str | None = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """
        Refresh all indexes with incremental updates.

        Args:
            tag: Index tag for this refresh operation
            current_files: Current file state
            read_file: Function to read file contents
            repo_name: Optional repository name

        Yields:
            Progress updates during indexing
        """
        if not self.indexes:
            yield IndexingProgressUpdate(
                progress=1.0, description="No indexes configured", status="done"
            )
            return

        # Acquire lock to prevent concurrent indexing
        directories = [tag.directory]
        if not await self.lock.acquire(directories):
            yield IndexingProgressUpdate(
                progress=0.0, description="Failed to acquire indexing lock", status="failed"
            )
            return

        try:
            # Calculate total expected time for progress tracking
            total_expected_time = sum(idx.relative_expected_time for idx in self.indexes)
            completed_time = 0.0

            for index in self.indexes:
                index_tag = IndexTag(
                    directory=tag.directory, branch=tag.branch, artifact_id=index.artifact_id
                )

                yield IndexingProgressUpdate(
                    progress=completed_time / total_expected_time,
                    description=f"Planning updates for {index.artifact_id}",
                    status="indexing",
                )

                try:
                    # Get or create circuit breaker for this index
                    if index.artifact_id not in self._circuit_breakers:
                        self._circuit_breakers[index.artifact_id] = CircuitBreaker(
                            failure_threshold=3,
                            recovery_timeout=30.0,
                            success_threshold=2,
                        )
                    breaker = self._circuit_breakers[index.artifact_id]

                    # Check if circuit breaker allows this operation
                    if not breaker.can_execute():
                        logger.warning(
                            f"Circuit breaker open for index {index.artifact_id}, skipping"
                        )
                        completed_time += index.relative_expected_time
                        yield IndexingProgressUpdate(
                            progress=completed_time / total_expected_time,
                            description=f"Skipped {index.artifact_id} (circuit breaker open)",
                            status="indexing",
                            warnings=[f"{index.artifact_id}: circuit breaker open"],
                        )
                        continue

                    # Calculate what needs to be updated for this index
                    from ...analysis.content_addressing import ContentAddressedIndexer

                    indexer = ContentAddressedIndexer(self.config)
                    refresh_results = await indexer.calculate_refresh_results(
                        index_tag, current_files, read_file
                    )

                    # Create mark_complete callback for this index
                    def mark_complete_wrapper(
                        items: list[PathAndCacheKey],
                        result_type: str,
                    ) -> None:
                        # Schedule async operations to run in the background
                        async def _async_mark_complete() -> None:
                            await indexer.mark_complete(items, result_type, index_tag)

                            # Update global cache
                            for item in items:
                                if result_type == "compute":
                                    # Store computed content in global cache for cross-branch reuse
                                    await self.global_cache.store_cached_content(
                                        content_hash=item.cache_key,
                                        artifact_id=index.artifact_id,
                                        content={"path": item.path, "cache_key": item.cache_key},
                                        tags=[index_tag],
                                    )
                                elif result_type == "delete":
                                    await self.global_cache.remove_tag(
                                        item.cache_key, index.artifact_id, index_tag
                                    )

                        # Create task but don't await it (fire and forget)
                        asyncio.create_task(_async_mark_complete())

                    # Update this index
                    async for update in index.update(
                        index_tag, refresh_results, mark_complete_wrapper, repo_name
                    ):
                        # Scale progress to overall progress
                        overall_progress = (
                            completed_time + (update.progress * index.relative_expected_time)
                        ) / total_expected_time

                        yield IndexingProgressUpdate(
                            progress=overall_progress,
                            description=f"{index.artifact_id}: {update.description}",
                            status=update.status,
                            warnings=update.warnings,
                            debug_info=update.debug_info,
                        )

                    completed_time += index.relative_expected_time
                    breaker.record_success()

                except Exception as e:
                    logger.error(f"Error updating index {index.artifact_id}: {e}")
                    self.error_collector.add_error(e, file_path=Path(index.artifact_id))
                    breaker.record_failure()

                    # Continue with other indexes
                    completed_time += index.relative_expected_time
                    yield IndexingProgressUpdate(
                        progress=completed_time / total_expected_time,
                        description=f"Error in {index.artifact_id}: {str(e)}",
                        status="indexing",
                        warnings=[f"{index.artifact_id}: {str(e)}"],
                    )

            # Cleanup orphaned content from global cache
            try:
                orphaned_count = await self.global_cache.cleanup_orphaned_content()
                if orphaned_count > 0:
                    logger.info(f"Cleaned up {orphaned_count} orphaned cache entries")
            except Exception as e:
                logger.warning(f"Error during orphaned content cleanup: {e}")

            # Final completion
            yield IndexingProgressUpdate(
                progress=1.0,
                description="Indexing complete",
                status="done",
                warnings=(
                    [str(error) for error in self.error_collector.errors]
                    if self.error_collector.errors
                    else None
                ),
            )

        finally:
            await self.lock.release()

    async def build_content_addresses(
        self, file_paths: list[str]
    ) -> dict[str, ContentAddress]:
        """
        Build content addresses for a list of files using ContentAddress.from_file().

        This provides SHA256-based content tracking for efficient change detection
        and cross-branch caching.

        Args:
            file_paths: List of file paths to build content addresses for

        Returns:
            Dictionary mapping file paths to their ContentAddress objects
        """
        addresses: dict[str, ContentAddress] = {}

        for path in file_paths:
            try:
                address = await ContentAddress.from_file(path)
                addresses[path] = address
            except Exception as e:
                logger.warning(f"Failed to build content address for {path}: {e}")

        return addresses

    async def cleanup_cache(self) -> int:
        """
        Cleanup orphaned content from the global cache.

        Returns:
            Number of orphaned entries removed
        """
        return await self.global_cache.cleanup_orphaned_content()
