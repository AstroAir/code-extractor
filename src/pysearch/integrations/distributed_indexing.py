"""
Distributed indexing system for large codebases.

This module implements parallel and distributed indexing capabilities
that can scale across multiple processes and machines for handling
very large codebases efficiently.

Classes:
    IndexingWorker: Worker process for distributed indexing
    IndexingCoordinator: Coordinates distributed indexing operations
    WorkQueue: Thread-safe work queue for indexing tasks
    DistributedIndexingEngine: Main distributed indexing engine

Features:
    - Multi-process parallel indexing
    - Work queue with priority scheduling
    - Load balancing across workers
    - Progress aggregation and monitoring
    - Fault tolerance and recovery
    - Resource usage monitoring
    - Dynamic worker scaling

Example:
    Basic distributed indexing:
        >>> from pysearch.distributed_indexing import DistributedIndexingEngine
        >>> engine = DistributedIndexingEngine(config, num_workers=4)
        >>> await engine.index_codebase(directories)

    Advanced usage with custom workers:
        >>> coordinator = IndexingCoordinator(config)
        >>> coordinator.add_worker_pool("embeddings", 2)
        >>> coordinator.add_worker_pool("parsing", 4)
        >>> await coordinator.process_work_queue()
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.config import SearchConfig
from ..analysis.content_addressing import IndexingProgressUpdate
from ..indexing.advanced.engine import IndexingEngine
from ..utils.logging_config import get_logger

logger = get_logger()


class WorkItemType(str, Enum):
    """Types of work items in the indexing queue."""

    FILE_DISCOVERY = "file_discovery"
    CONTENT_EXTRACTION = "content_extraction"
    ENTITY_PARSING = "entity_parsing"
    CHUNKING = "chunking"
    EMBEDDING_GENERATION = "embedding_generation"
    INDEX_UPDATE = "index_update"


@dataclass
class WorkItem:
    """Represents a unit of work in the indexing pipeline."""

    item_id: str
    item_type: WorkItemType
    priority: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    worker_id: str | None = None
    error: str | None = None


@dataclass
class WorkerStats:
    """Statistics for an indexing worker."""

    worker_id: str
    process_id: int
    items_processed: int = 0
    items_failed: int = 0
    total_processing_time: float = 0.0
    current_item: str | None = None
    last_heartbeat: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class WorkQueue:
    """
    Thread-safe work queue for distributed indexing.

    Manages work items with priority scheduling, dependency tracking,
    and progress monitoring across multiple workers.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._pending: dict[str, WorkItem] = {}
        self._completed: dict[str, WorkItem] = {}
        self._failed: dict[str, WorkItem] = {}
        self._lock = asyncio.Lock()

    async def add_work_item(self, item: WorkItem) -> bool:
        """Add a work item to the queue."""
        async with self._lock:
            try:
                # Check dependencies
                for dep_id in item.dependencies:
                    if dep_id not in self._completed:
                        logger.debug(f"Work item {item.item_id} waiting for dependency {dep_id}")
                        return False

                # Add to queue (priority queue uses negative priority for max-heap behavior)
                self._queue.put((-item.priority, item.created_at, item))
                self._pending[item.item_id] = item
                return True

            except queue.Full:
                logger.warning(f"Work queue full, cannot add item {item.item_id}")
                return False

    async def get_work_item(self, timeout: float = 1.0) -> WorkItem | None:
        """Get the next work item from the queue."""
        try:
            # Use non-blocking get to avoid blocking the event loop
            _, _, item = self._queue.get_nowait()
            async with self._lock:
                if item.item_id in self._pending:
                    item.started_at = time.time()
                    return item
            return None
        except queue.Empty:
            return None

    async def complete_work_item(self, item_id: str, result: Any = None) -> None:
        """Mark a work item as completed."""
        async with self._lock:
            if item_id in self._pending:
                item = self._pending.pop(item_id)
                item.completed_at = time.time()
                self._completed[item_id] = item

                # Check if any pending items can now be processed
                await self._check_dependencies()

    async def fail_work_item(self, item_id: str, error: str) -> None:
        """Mark a work item as failed."""
        async with self._lock:
            if item_id in self._pending:
                item = self._pending.pop(item_id)
                item.error = error
                self._failed[item_id] = item

    async def _check_dependencies(self) -> None:
        """Check if any pending items can be processed due to completed dependencies."""
        items_to_enqueue: list[WorkItem] = []
        for item_id, item in list(self._pending.items()):
            if not item.dependencies:
                continue
            if all(dep_id in self._completed for dep_id in item.dependencies):
                items_to_enqueue.append(item)

        for item in items_to_enqueue:
            try:
                self._queue.put_nowait((-item.priority, item.created_at, item))
            except queue.Full:
                logger.warning(f"Queue full, cannot re-enqueue item {item.item_id}")

    def get_stats(self) -> dict[str, Any]:
        """Get work queue statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "pending_items": len(self._pending),
            "completed_items": len(self._completed),
            "failed_items": len(self._failed),
        }


class IndexingWorker:
    """
    Worker process for distributed indexing operations.

    Each worker handles specific types of indexing work and reports
    progress back to the coordinator.
    """

    def __init__(
        self,
        worker_id: str,
        config: SearchConfig,
        work_types: list[WorkItemType],
    ):
        self.worker_id = worker_id
        self.config = config
        self.work_types = work_types
        self.stats = WorkerStats(worker_id=worker_id, process_id=mp.current_process().pid)
        self.running = False

        # Initialize indexing engine for this worker
        self.indexing_engine = IndexingEngine(config)

    async def start(self, work_queue: WorkQueue) -> None:
        """Start the worker process."""
        self.running = True
        logger.info(f"Worker {self.worker_id} started")

        await self.indexing_engine.initialize()

        while self.running:
            try:
                # Get work item
                item = await work_queue.get_work_item(timeout=1.0)

                if item is None:
                    # No work available, update heartbeat and continue
                    self.stats.last_heartbeat = time.time()
                    await asyncio.sleep(0.1)
                    continue

                # Check if we can handle this work type
                if item.item_type not in self.work_types:
                    logger.warning(f"Worker {self.worker_id} cannot handle {item.item_type}")
                    await work_queue.fail_work_item(item.item_id, "Unsupported work type")
                    continue

                # Process work item
                self.stats.current_item = item.item_id
                start_time = time.time()

                try:
                    await self._process_work_item(item)
                    await work_queue.complete_work_item(item.item_id)
                    self.stats.items_processed += 1

                except Exception as e:
                    logger.error(f"Worker {self.worker_id} failed to process {item.item_id}: {e}")
                    await work_queue.fail_work_item(item.item_id, str(e))
                    self.stats.items_failed += 1

                finally:
                    processing_time = time.time() - start_time
                    self.stats.total_processing_time += processing_time
                    self.stats.current_item = None
                    self.stats.last_heartbeat = time.time()

                    # Update resource usage
                    await self._update_resource_usage()

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(1.0)

    async def _process_work_item(self, item: WorkItem) -> None:
        """Process a specific work item."""
        if item.item_type == WorkItemType.FILE_DISCOVERY:
            await self._process_file_discovery(item)
        elif item.item_type == WorkItemType.CONTENT_EXTRACTION:
            await self._process_content_extraction(item)
        elif item.item_type == WorkItemType.ENTITY_PARSING:
            await self._process_entity_parsing(item)
        elif item.item_type == WorkItemType.CHUNKING:
            await self._process_chunking(item)
        elif item.item_type == WorkItemType.EMBEDDING_GENERATION:
            await self._process_embedding_generation(item)
        elif item.item_type == WorkItemType.INDEX_UPDATE:
            await self._process_index_update(item)
        else:
            raise ValueError(f"Unknown work item type: {item.item_type}")

    async def _process_file_discovery(self, item: WorkItem) -> None:
        """Process file discovery work item.

        Discovers files in the given directory using the project's iter_files utility,
        respecting include/exclude patterns and language filters from config.
        Discovered files are stored in item.data["discovered_files"].
        """
        from pathlib import Path

        from ..utils.utils import file_meta, iter_files

        directory = item.data["directory"]
        logger.debug(f"Worker {self.worker_id} discovering files in {directory}")

        discovered: list[str] = []
        for file_path in iter_files(
            roots=[directory],
            include=self.config.get_include_patterns(),
            exclude=self.config.get_exclude_patterns(),
            follow_symlinks=self.config.follow_symlinks,
            prune_excluded_dirs=self.config.dir_prune_exclude,
            language_filter=self.config.languages,
        ):
            try:
                meta = file_meta(file_path)
                if meta and meta.size <= self.config.file_size_limit:
                    discovered.append(str(file_path))
            except Exception as e:
                logger.debug(f"Skipping {file_path}: {e}")

        item.data["discovered_files"] = discovered
        logger.info(
            f"Worker {self.worker_id} discovered {len(discovered)} files in {directory}"
        )

    async def _process_content_extraction(self, item: WorkItem) -> None:
        """Process content extraction work item.

        Reads file content safely, storing the text in item.data["content"].
        """
        from pathlib import Path

        from ..utils.utils import read_text_safely

        file_path = item.data["file_path"]
        logger.debug(f"Worker {self.worker_id} extracting content from {file_path}")

        content = read_text_safely(Path(file_path))
        if content is None:
            raise ValueError(f"Cannot read file: {file_path}")

        item.data["content"] = content
        item.data["line_count"] = content.count("\n") + 1

    async def _process_entity_parsing(self, item: WorkItem) -> None:
        """Process entity parsing work item.

        Parses code entities (functions, classes, imports) from file content
        using the project's language detection and AST parsing capabilities.
        """
        from pathlib import Path

        from ..analysis.language_detection import detect_language

        file_path = item.data["file_path"]
        content = item.data.get("content", "")
        logger.debug(f"Worker {self.worker_id} parsing entities from {file_path}")

        language = detect_language(Path(file_path))
        entities: list[dict[str, Any]] = []

        try:
            from ..search.matchers import extract_ast_entities

            raw_entities = extract_ast_entities(Path(file_path), content, language)
            entities = [
                {"name": e.name, "type": e.entity_type, "line": e.line}
                for e in raw_entities
            ] if raw_entities else []
        except (ImportError, Exception) as e:
            logger.debug(f"AST parsing not available for {file_path}: {e}")

        item.data["entities"] = entities
        item.data["language"] = language.value if language else "unknown"

    async def _process_chunking(self, item: WorkItem) -> None:
        """Process chunking work item.

        Splits file content into chunks using the configured chunking strategy.
        """
        file_path = item.data["file_path"]
        content = item.data.get("content", "")
        logger.debug(f"Worker {self.worker_id} chunking {file_path}")

        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        lines = content.split("\n")
        chunks: list[dict[str, Any]] = []

        # Simple line-based chunking with overlap
        start = 0
        chunk_idx = 0
        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            chunk_content = "\n".join(lines[start:end])
            chunks.append({
                "chunk_id": f"{file_path}:chunk_{chunk_idx}",
                "content": chunk_content,
                "start_line": start + 1,
                "end_line": end,
                "file_path": file_path,
            })
            chunk_idx += 1
            start = end - chunk_overlap if end < len(lines) else end

        item.data["chunks"] = chunks
        logger.debug(f"Worker {self.worker_id} created {len(chunks)} chunks for {file_path}")

    async def _process_embedding_generation(self, item: WorkItem) -> None:
        """Process embedding generation work item.

        Delegates to the indexing engine's coordinator for embedding generation
        if vector indexing is enabled, otherwise stores placeholder metadata.
        """
        chunks = item.data.get("chunks", [])
        logger.debug(
            f"Worker {self.worker_id} generating embeddings for {len(chunks)} chunks"
        )

        # Delegate to the indexing engine if it supports embedding
        if hasattr(self.indexing_engine, "coordinator"):
            for idx in self.indexing_engine.coordinator.indexes:
                if hasattr(idx, "add_chunks"):
                    try:
                        await idx.add_chunks(chunks)
                    except Exception as e:
                        logger.warning(f"Index {idx} failed to add chunks: {e}")

        item.data["embeddings_generated"] = len(chunks)

    async def _process_index_update(self, item: WorkItem) -> None:
        """Process index update work item.

        Triggers a targeted index refresh for specific files or the full index.
        """
        index_type = item.data.get("index_type", "all")
        file_path = item.data.get("file_path")
        directory = item.data.get("directory", ".")
        logger.debug(f"Worker {self.worker_id} updating {index_type} index")

        if file_path:
            await self.indexing_engine.refresh_file(
                file_path=file_path,
                directory=directory,
            )
        else:
            async for _update in self.indexing_engine.refresh_index(
                directories=[directory]
            ):
                pass  # consume progress updates silently

    async def _update_resource_usage(self) -> None:
        """Update resource usage statistics."""
        try:
            import psutil

            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.stats.cpu_usage_percent = process.cpu_percent()
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.debug(f"Error updating resource usage: {e}")

    def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info(f"Worker {self.worker_id} stopped")


class DistributedIndexingEngine:
    """
    Main distributed indexing engine.

    Coordinates multiple workers to process large codebases efficiently
    with parallel processing, load balancing, and progress monitoring.
    """

    def __init__(
        self,
        config: SearchConfig,
        num_workers: int | None = None,
        max_queue_size: int = 10000,
    ):
        self.config = config
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.work_queue = WorkQueue(max_queue_size)
        self.workers: list[IndexingWorker] = []
        self.worker_tasks: list[asyncio.Task] = []
        self.running = False

    async def start_workers(self) -> None:
        """Start all indexing workers."""
        if self.running:
            return

        self.running = True

        # Create workers with different specializations
        for i in range(self.num_workers):
            if i < self.num_workers // 2:
                # First half: file processing workers
                work_types = [
                    WorkItemType.FILE_DISCOVERY,
                    WorkItemType.CONTENT_EXTRACTION,
                    WorkItemType.ENTITY_PARSING,
                    WorkItemType.CHUNKING,
                ]
            else:
                # Second half: indexing workers
                work_types = [
                    WorkItemType.EMBEDDING_GENERATION,
                    WorkItemType.INDEX_UPDATE,
                ]

            worker = IndexingWorker(
                worker_id=f"worker_{i}",
                config=self.config,
                work_types=work_types,
            )

            self.workers.append(worker)

            # Start worker task
            task = asyncio.create_task(worker.start(self.work_queue))
            self.worker_tasks.append(task)

        logger.info(f"Started {len(self.workers)} indexing workers")

    async def stop_workers(self) -> None:
        """Stop all indexing workers."""
        if not self.running:
            return

        self.running = False

        # Stop all workers
        for worker in self.workers:
            worker.stop()

        # Wait for worker tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self.workers.clear()
        self.worker_tasks.clear()

        logger.info("All indexing workers stopped")

    async def index_codebase(
        self,
        directories: list[str],
        branch: str | None = None,
        repo_name: str | None = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """
        Index codebase using distributed workers.

        Args:
            directories: Directories to index
            branch: Git branch name
            repo_name: Repository name

        Yields:
            Progress updates during indexing
        """
        # Start workers
        await self.start_workers()

        try:
            # Create work items for file discovery
            for directory in directories:
                work_item = WorkItem(
                    item_id=f"discover_{directory}",
                    item_type=WorkItemType.FILE_DISCOVERY,
                    priority=10,  # High priority for discovery
                    data={"directory": directory},
                )
                await self.work_queue.add_work_item(work_item)

            # Monitor progress
            start_time = time.time()
            last_update = 0.0

            while True:
                # Get queue stats
                stats = self.work_queue.get_stats()

                # Calculate progress
                total_items = (
                    stats["completed_items"] + stats["failed_items"] + stats["pending_items"]
                )
                if total_items > 0:
                    progress = (stats["completed_items"] + stats["failed_items"]) / total_items
                else:
                    progress = 0.0

                # Update progress periodically
                current_time = time.time()
                if current_time - last_update >= 1.0:  # Update every second
                    description = (
                        f"Processing {stats['pending_items']} items "
                        f"({stats['completed_items']} completed, {stats['failed_items']} failed)"
                    )
                    yield IndexingProgressUpdate(
                        progress=progress,
                        description=description,
                        status="indexing",
                        debug_info=f"Workers: {len(self.workers)}, Queue: {stats['queue_size']}",
                    )
                    last_update = current_time

                # Check if done
                if stats["pending_items"] == 0 and stats["queue_size"] == 0:
                    break

                # Check for timeout
                if current_time - start_time > 3600:  # 1 hour timeout
                    logger.warning("Distributed indexing timeout")
                    break

                await asyncio.sleep(0.1)

            # Final progress update
            final_stats = self.work_queue.get_stats()
            final_description = (
                f"Distributed indexing complete: {final_stats['completed_items']} items processed, "
                f"{final_stats['failed_items']} failed"
            )
            yield IndexingProgressUpdate(
                progress=1.0,
                description=final_description,
                status="done" if final_stats["failed_items"] == 0 else "done_with_errors",
            )

        finally:
            await self.stop_workers()

    async def get_worker_stats(self) -> list[WorkerStats]:
        """Get statistics for all workers."""
        return [worker.stats for worker in self.workers]

    async def scale_workers(self, target_count: int) -> None:
        """Dynamically scale the number of workers."""
        current_count = len(self.workers)

        if target_count > current_count:
            # Add workers
            for i in range(current_count, target_count):
                worker = IndexingWorker(
                    worker_id=f"worker_{i}",
                    config=self.config,
                    work_types=list(WorkItemType),  # All work types
                )

                self.workers.append(worker)
                task = asyncio.create_task(worker.start(self.work_queue))
                self.worker_tasks.append(task)

            logger.info(f"Scaled up to {target_count} workers")

        elif target_count < current_count:
            # Remove workers
            workers_to_remove = self.workers[target_count:]
            tasks_to_cancel = self.worker_tasks[target_count:]

            # Stop excess workers
            for worker in workers_to_remove:
                worker.stop()

            # Cancel their tasks
            for task in tasks_to_cancel:
                task.cancel()

            # Update lists
            self.workers = self.workers[:target_count]
            self.worker_tasks = self.worker_tasks[:target_count]

            logger.info(f"Scaled down to {target_count} workers")

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        worker_stats = await self.get_worker_stats()
        queue_stats = self.work_queue.get_stats()

        # Aggregate worker metrics
        total_processed = sum(w.items_processed for w in worker_stats)
        total_failed = sum(w.items_failed for w in worker_stats)
        avg_processing_time = (
            sum(w.total_processing_time for w in worker_stats) / len(worker_stats)
            if worker_stats
            else 0.0
        )
        total_memory_mb = sum(w.memory_usage_mb for w in worker_stats)
        avg_cpu_percent = (
            sum(w.cpu_usage_percent for w in worker_stats) / len(worker_stats)
            if worker_stats
            else 0.0
        )

        return {
            "workers": {
                "total_workers": len(worker_stats),
                "active_workers": len([w for w in worker_stats if w.current_item]),
                "total_items_processed": total_processed,
                "total_items_failed": total_failed,
                "average_processing_time": avg_processing_time,
                "total_memory_usage_mb": total_memory_mb,
                "average_cpu_usage_percent": avg_cpu_percent,
            },
            "queue": queue_stats,
            "performance": {
                "throughput_items_per_second": total_processed / max(avg_processing_time, 1.0),
                "error_rate": total_failed / max(total_processed + total_failed, 1),
                "memory_per_worker_mb": (
                    total_memory_mb / len(worker_stats) if worker_stats else 0.0
                ),
            },
        }
