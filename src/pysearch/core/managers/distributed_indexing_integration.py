"""
Distributed indexing integration manager.

This module provides distributed indexing capabilities for pysearch,
enabling parallel and distributed indexing across multiple workers
for handling very large codebases efficiently.

Classes:
    DistributedIndexingManager: Manages distributed indexing functionality

Key Features:
    - Multi-worker parallel indexing
    - Dynamic worker scaling
    - Worker statistics and performance monitoring
    - Progress tracking during distributed indexing

Example:
    Using distributed indexing:
        >>> from pysearch.core.managers.distributed_indexing_integration import DistributedIndexingManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = DistributedIndexingManager(config)
        >>> manager.enable_distributed_indexing(num_workers=4)
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..config import SearchConfig


class DistributedIndexingManager:
    """Manages distributed indexing functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._engine: Any = None
        self._enabled = False

    def enable_distributed_indexing(
        self, num_workers: int | None = None, max_queue_size: int = 10000
    ) -> bool:
        """
        Enable distributed indexing.

        Args:
            num_workers: Number of worker processes (defaults to min(cpu_count, 8))
            max_queue_size: Maximum work queue size

        Returns:
            True if distributed indexing was enabled successfully, False otherwise
        """
        if self._enabled:
            return True

        try:
            from ...integrations.distributed_indexing import DistributedIndexingEngine

            self._engine = DistributedIndexingEngine(
                config=self.config,
                num_workers=num_workers,
                max_queue_size=max_queue_size,
            )
            self._enabled = True
            return True

        except Exception:
            return False

    def disable_distributed_indexing(self) -> None:
        """Disable distributed indexing and stop all workers."""
        if not self._enabled:
            return

        if self._engine:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._engine.stop_workers())
                else:
                    loop.run_until_complete(self._engine.stop_workers())
            except RuntimeError:
                # No event loop or can't run in current context
                pass

        self._engine = None
        self._enabled = False

    def is_distributed_enabled(self) -> bool:
        """Check if distributed indexing is enabled."""
        return self._enabled

    async def index_codebase(
        self,
        directories: list[str],
        branch: str | None = None,
        repo_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Index codebase using distributed workers.

        Args:
            directories: Directories to index
            branch: Git branch name
            repo_name: Repository name

        Returns:
            List of progress update dictionaries
        """
        if not self._engine:
            return []

        try:
            updates: list[dict[str, Any]] = []
            async for update in self._engine.index_codebase(
                directories=directories,
                branch=branch,
                repo_name=repo_name,
            ):
                updates.append(
                    {
                        "progress": update.progress,
                        "description": update.description,
                        "status": update.status,
                        "debug_info": getattr(update, "debug_info", ""),
                    }
                )
            return updates
        except Exception:
            return []

    async def get_worker_stats(self) -> list[dict[str, Any]]:
        """
        Get statistics for all workers.

        Returns:
            List of worker statistics dictionaries
        """
        if not self._engine:
            return []

        try:
            stats = await self._engine.get_worker_stats()
            return [
                {
                    "worker_id": s.worker_id,
                    "process_id": s.process_id,
                    "items_processed": s.items_processed,
                    "items_failed": s.items_failed,
                    "total_processing_time": s.total_processing_time,
                    "current_item": s.current_item,
                    "memory_usage_mb": s.memory_usage_mb,
                    "cpu_usage_percent": s.cpu_usage_percent,
                }
                for s in stats
            ]
        except Exception:
            return []

    async def scale_workers(self, target_count: int) -> bool:
        """
        Dynamically scale the number of workers.

        Args:
            target_count: Target number of workers

        Returns:
            True if scaling succeeded, False otherwise
        """
        if not self._engine:
            return False

        try:
            await self._engine.scale_workers(target_count)
            return True
        except Exception:
            return False

    async def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dictionary with worker, queue, and performance metrics
        """
        if not self._engine:
            return {}

        try:
            return await self._engine.get_performance_metrics()
        except Exception:
            return {}

    def get_queue_stats(self) -> dict[str, Any]:
        """
        Get work queue statistics (synchronous).

        Returns:
            Dictionary with queue statistics
        """
        if not self._engine:
            return {}

        try:
            return self._engine.work_queue.get_stats()
        except Exception:
            return {}
