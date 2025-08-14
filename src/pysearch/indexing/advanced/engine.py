"""
Enhanced code indexing engine with tag-based management and content addressing.

This module implements the core enhanced indexing engine that coordinates multiple
index types using Continue's tag-based approach with content addressing for
efficient incremental updates and cross-branch caching.

Classes:
    EnhancedCodebaseIndex: Abstract base for all index types
    IndexCoordinator: Coordinates multiple index types
    EnhancedIndexingEngine: Main indexing engine
    IndexLock: Prevents concurrent indexing operations

Features:
    - Tag-based index management (directory + branch + artifact)
    - Content-addressed caching with SHA256 hashes
    - Multiple index types (snippets, full-text, chunks, vectors)
    - Incremental updates with smart diffing
    - Global cache for cross-branch content sharing
    - Batch processing for memory efficiency
    - Progress tracking with pause/resume capability
    - Robust error handling and recovery

Example:
    Basic enhanced indexing:
        >>> from pysearch.enhanced_indexing_engine import EnhancedIndexingEngine
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["./src"])
        >>> engine = EnhancedIndexingEngine(config)
        >>> await engine.refresh_index()

    Advanced usage with custom indexes:
        >>> from pysearch.enhanced_indexing_engine import IndexCoordinator
        >>> coordinator = IndexCoordinator(config)
        >>> coordinator.add_index(CustomCodebaseIndex())
        >>> await coordinator.refresh_all_indexes()
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from ...core.config import SearchConfig
from ...analysis.content_addressing import (
    ContentAddress,
    GlobalCacheManager,
    IndexTag,
    IndexingProgressUpdate,
    MarkCompleteCallback,
    PathAndCacheKey,
    RefreshIndexResults,
)
from ...utils.error_handling import ErrorCollector
from ...utils.logging_config import get_logger
from ...utils.utils import iter_files

logger = get_logger()


class EnhancedCodebaseIndex(ABC):
    """
    Abstract base class for all enhanced index types.

    This interface defines the contract that all index implementations must follow
    to participate in the enhanced indexing system.
    """

    @property
    @abstractmethod
    def artifact_id(self) -> str:
        """Unique identifier for this index type."""
        pass

    @property
    @abstractmethod
    def relative_expected_time(self) -> float:
        """Relative time cost for this index type (1.0 = baseline)."""
        pass

    @abstractmethod
    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: Optional[str] = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """
        Update the index with new/changed/deleted files.

        Args:
            tag: Index tag identifying the specific index instance
            results: Files to compute/delete/add_tag/remove_tag
            mark_complete: Callback to mark operations as complete
            repo_name: Optional repository name for context

        Yields:
            Progress updates during the indexing operation
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        tag: IndexTag,
        limit: int = 50,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Retrieve results from this index.

        Args:
            query: Search query
            tag: Index tag to search within
            limit: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        pass


class IndexLock:
    """
    Prevents concurrent indexing operations across multiple processes.

    Uses file-based locking to coordinate indexing operations and prevent
    SQLite concurrent write errors.
    """

    def __init__(self, cache_dir: Path):
        self.lock_file = cache_dir / "indexing.lock"
        self.cache_dir = cache_dir

    async def acquire(self, directories: List[str], timeout: float = 300.0) -> bool:
        """
        Acquire indexing lock.

        Args:
            directories: List of directories being indexed
            timeout: Maximum time to wait for lock

        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if not self.lock_file.exists():
                    # Create lock file
                    lock_data = {
                        "directories": directories,
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                    }

                    # Atomic write
                    temp_file = self.lock_file.with_suffix(".tmp")
                    temp_file.write_text(str(lock_data))
                    temp_file.rename(self.lock_file)

                    return True
                else:
                    # Check if existing lock is stale
                    try:
                        lock_data = eval(self.lock_file.read_text())
                        if time.time() - lock_data["timestamp"] > 600:  # 10 minutes
                            logger.warning("Removing stale indexing lock")
                            self.lock_file.unlink()
                            continue
                    except Exception:
                        # Corrupted lock file, remove it
                        self.lock_file.unlink()
                        continue

                # Wait before retrying
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error acquiring index lock: {e}")
                await asyncio.sleep(1.0)

        return False

    async def release(self) -> None:
        """Release the indexing lock."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.error(f"Error releasing index lock: {e}")

    async def update_timestamp(self) -> None:
        """Update lock timestamp to prevent stale lock detection."""
        try:
            if self.lock_file.exists():
                lock_data = eval(self.lock_file.read_text())
                lock_data["timestamp"] = time.time()
                self.lock_file.write_text(str(lock_data))
        except Exception as e:
            logger.error(f"Error updating lock timestamp: {e}")


class IndexCoordinator:
    """
    Coordinates multiple index types for comprehensive code indexing.

    This class manages the lifecycle of multiple index implementations,
    ensuring they work together efficiently and handle updates correctly.
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.indexes: List[EnhancedCodebaseIndex] = []
        self.global_cache = GlobalCacheManager(config.resolve_cache_dir())
        self.error_collector = ErrorCollector()
        self.lock = IndexLock(config.resolve_cache_dir())

    def add_index(self, index: EnhancedCodebaseIndex) -> None:
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

    def get_index(self, artifact_id: str) -> Optional[EnhancedCodebaseIndex]:
        """Get an index by artifact ID."""
        for index in self.indexes:
            if index.artifact_id == artifact_id:
                return index
        return None

    async def refresh_all_indexes(
        self,
        tag: IndexTag,
        current_files: Dict[str, Any],
        read_file: Callable[[str], str],
        repo_name: Optional[str] = None,
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
                progress=1.0,
                description="No indexes configured",
                status="done"
            )
            return

        # Acquire lock to prevent concurrent indexing
        directories = [tag.directory]
        if not await self.lock.acquire(directories):
            yield IndexingProgressUpdate(
                progress=0.0,
                description="Failed to acquire indexing lock",
                status="failed"
            )
            return

        try:
            # Calculate total expected time for progress tracking
            total_expected_time = sum(idx.relative_expected_time for idx in self.indexes)
            completed_time = 0.0

            for index in self.indexes:
                index_tag = IndexTag(
                    directory=tag.directory,
                    branch=tag.branch,
                    artifact_id=index.artifact_id
                )

                yield IndexingProgressUpdate(
                    progress=completed_time / total_expected_time,
                    description=f"Planning updates for {index.artifact_id}",
                    status="indexing"
                )

                try:
                    # Calculate what needs to be updated for this index
                    from ...analysis.content_addressing import ContentAddressedIndexer
                    indexer = ContentAddressedIndexer(self.config)
                    refresh_results = await indexer.calculate_refresh_results(
                        index_tag, current_files, read_file
                    )

                    # Create mark_complete callback for this index
                    async def mark_complete_wrapper(
                        items: List[PathAndCacheKey],
                        result_type: str,
                    ) -> None:
                        await indexer.mark_complete(items, result_type, index_tag)

                        # Update global cache
                        for item in items:
                            if result_type == "compute":
                                # Store in global cache (implementation depends on index type)
                                pass
                            elif result_type == "delete":
                                await self.global_cache.remove_tag(
                                    item.cache_key, index.artifact_id, index_tag
                                )

                    # Update this index
                    index_progress = 0.0
                    async for update in index.update(
                        index_tag, refresh_results, mark_complete_wrapper, repo_name
                    ):
                        # Scale progress to overall progress
                        overall_progress = (
                            completed_time +
                            (update.progress * index.relative_expected_time)
                        ) / total_expected_time

                        yield IndexingProgressUpdate(
                            progress=overall_progress,
                            description=f"{index.artifact_id}: {update.description}",
                            status=update.status,
                            warnings=update.warnings,
                            debug_info=update.debug_info
                        )
                        index_progress = update.progress

                    completed_time += index.relative_expected_time

                except Exception as e:
                    logger.error(f"Error updating index {index.artifact_id}: {e}")
                    self.error_collector.add_error(index.artifact_id, str(e))

                    # Continue with other indexes
                    completed_time += index.relative_expected_time
                    yield IndexingProgressUpdate(
                        progress=completed_time / total_expected_time,
                        description=f"Error in {index.artifact_id}: {str(e)}",
                        status="indexing",
                        warnings=[f"{index.artifact_id}: {str(e)}"]
                    )

            # Final completion
            yield IndexingProgressUpdate(
                progress=1.0,
                description="Indexing complete",
                status="done",
                warnings=self.error_collector.get_all_errors() if self.error_collector.has_errors() else None
            )

        finally:
            await self.lock.release()


class EnhancedIndexingEngine:
    """
    Main enhanced indexing engine that coordinates all indexing operations.

    This engine provides the high-level interface for enhanced indexing operations,
    managing multiple index types, handling incremental updates, and providing
    comprehensive progress tracking.
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.coordinator = IndexCoordinator(config)
        self.error_collector = ErrorCollector()
        self._paused = False
        self._cancel_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the indexing engine and load default indexes."""
        # Load default indexes based on configuration
        await self._load_default_indexes()

    async def _load_default_indexes(self) -> None:
        """Load default index implementations."""
        # Import and add default indexes
        try:
            from ..indexes.code_snippets_index import EnhancedCodeSnippetsIndex
            self.coordinator.add_index(EnhancedCodeSnippetsIndex(self.config))
        except ImportError:
            logger.warning("Code snippets index not available")

        try:
            from ..indexes.full_text_index import EnhancedFullTextIndex
            self.coordinator.add_index(EnhancedFullTextIndex(self.config))
        except ImportError:
            logger.warning("Full text index not available")

        try:
            from ..indexes.chunk_index import EnhancedChunkIndex
            self.coordinator.add_index(EnhancedChunkIndex(self.config))
        except ImportError:
            logger.warning("Chunk index not available")

        if self.config.enable_enhanced_indexing:
            try:
                from ..indexes.vector_index import EnhancedVectorIndex
                self.coordinator.add_index(EnhancedVectorIndex(self.config))
            except ImportError:
                logger.warning("Vector index not available")

    async def refresh_index(
        self,
        directories: Optional[List[str]] = None,
        branch: Optional[str] = None,
        repo_name: Optional[str] = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """
        Refresh indexes for specified directories.

        Args:
            directories: Directories to index (defaults to config paths)
            branch: Git branch name (auto-detected if None)
            repo_name: Repository name (auto-detected if None)

        Yields:
            Progress updates during indexing
        """
        if directories is None:
            directories = self.config.paths

        # Auto-detect branch if not provided
        if branch is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=directories[0],
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                branch = result.stdout.strip() if result.returncode == 0 else "main"
            except Exception:
                branch = "main"

        # Auto-detect repo name if not provided
        if repo_name is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=directories[0],
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                if result.returncode == 0:
                    url = result.stdout.strip()
                    repo_name = Path(url).stem.replace(".git", "")
            except Exception:
                repo_name = Path(directories[0]).name

        for directory in directories:
            yield IndexingProgressUpdate(
                progress=0.0,
                description=f"Starting indexing for {directory}",
                status="loading"
            )

            # Discover files
            yield IndexingProgressUpdate(
                progress=0.1,
                description=f"Discovering files in {directory}",
                status="indexing"
            )

            current_files = {}
            file_count = 0

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
                        current_files[str(file_path)] = {
                            "size": meta.size,
                            "mtime": meta.mtime,
                        }
                        file_count += 1
                except Exception as e:
                    self.error_collector.add_error(str(file_path), str(e))

            yield IndexingProgressUpdate(
                progress=0.2,
                description=f"Found {file_count} files to process",
                status="indexing"
            )

            # Create tag for this indexing operation
            tag = IndexTag(
                directory=directory,
                branch=branch,
                artifact_id="*"  # Will be replaced per index
            )

            # Read file function
            async def read_file(path: str) -> str:
                return await read_text_safely(Path(path))

            # Refresh all indexes
            progress_offset = 0.2
            progress_scale = 0.8

            async for update in self.coordinator.refresh_all_indexes(
                tag, current_files, read_file, repo_name
            ):
                if self._paused:
                    yield IndexingProgressUpdate(
                        progress=update.progress,
                        description="Indexing paused",
                        status="paused"
                    )

                    while self._paused and not self._cancel_event.is_set():
                        await asyncio.sleep(0.1)

                    if self._cancel_event.is_set():
                        yield IndexingProgressUpdate(
                            progress=update.progress,
                            description="Indexing cancelled",
                            status="cancelled"
                        )
                        return

                # Scale progress to account for file discovery
                scaled_progress = progress_offset + (update.progress * progress_scale)

                yield IndexingProgressUpdate(
                    progress=scaled_progress,
                    description=update.description,
                    status=update.status,
                    warnings=update.warnings,
                    debug_info=update.debug_info
                )

    async def refresh_file(
        self,
        file_path: str,
        directory: str,
        branch: Optional[str] = None,
        repo_name: Optional[str] = None,
    ) -> None:
        """
        Refresh indexes for a single file.

        Args:
            file_path: Path to the file to refresh
            directory: Directory containing the file
            branch: Git branch name
            repo_name: Repository name
        """
        if branch is None:
            branch = "main"  # Default branch

        try:
            # Create file stats
            meta = file_meta(Path(file_path))
            if meta is None or meta.size > self.config.file_size_limit:
                return

            current_files = {
                file_path: {
                    "size": meta.size,
                    "mtime": meta.mtime,
                }
            }

            # Create tag
            tag = IndexTag(
                directory=directory,
                branch=branch,
                artifact_id="*"
            )

            # Read file function
            async def read_file(path: str) -> str:
                return await read_text_safely(Path(path))

            # Refresh all indexes for this file
            async for update in self.coordinator.refresh_all_indexes(
                tag, current_files, read_file, repo_name
            ):
                # Log progress for single file updates
                logger.debug(f"File refresh progress: {update.description}")

        except Exception as e:
            logger.error(f"Error refreshing file {file_path}: {e}")
            self.error_collector.add_error(file_path, str(e))

    def pause(self) -> None:
        """Pause indexing operations."""
        self._paused = True
        logger.info("Indexing paused")

    def resume(self) -> None:
        """Resume indexing operations."""
        self._paused = False
        logger.info("Indexing resumed")

    def cancel(self) -> None:
        """Cancel indexing operations."""
        self._cancel_event.set()
        logger.info("Indexing cancelled")

    @property
    def is_paused(self) -> bool:
        """Check if indexing is paused."""
        return self._paused

    @property
    def is_cancelled(self) -> bool:
        """Check if indexing is cancelled."""
        return self._cancel_event.is_set()

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for all indexes."""
        stats = {
            "total_indexes": len(self.indexes),
            "index_types": [idx.artifact_id for idx in self.indexes],
            "cache_dir": str(self.config.resolve_cache_dir()),
            "errors": self.error_collector.get_all_errors(),
        }

        # Add global cache stats
        try:
            cache_stats = await self._get_cache_stats()
            stats["cache"] = cache_stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")

        return stats

    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        # This would query the global cache database for statistics
        # Implementation depends on GlobalCacheManager
        return {
            "total_entries": 0,
            "total_size_bytes": 0,
            "hit_rate": 0.0,
        }
