"""
Enhanced indexing engine for coordinating all indexing operations.

This module provides the main IndexingEngine class that serves as the
high-level interface for enhanced indexing operations, managing multiple index
types and providing comprehensive progress tracking.

Classes:
    IndexingEngine: Main indexing engine

Features:
    - High-level indexing interface
    - Multiple index type coordination
    - Progress tracking with pause/resume capability
    - Automatic index discovery and loading
    - Error handling and recovery
    - File discovery and processing

Example:
    Basic enhanced indexing:
        >>> from pysearch.indexing.advanced.engine import IndexingEngine
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["./src"])
        >>> engine = IndexingEngine(config)
        >>> await engine.refresh_index()
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from ...analysis.content_addressing import IndexingProgressUpdate, IndexTag
from ...core.config import SearchConfig
from ...utils.error_handling import (
    EnhancedErrorHandler,
    ErrorCollector,
    ErrorSeverity,
)
from ...utils.helpers import file_meta, iter_files, read_text_safely
from ...utils.logging_config import get_logger
from ...utils.performance_monitoring import MetricsCollector, PerformanceProfiler
from .coordinator import IndexCoordinator

logger = get_logger()


class IndexingEngine:
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

        # Advanced error handling with recovery strategies
        self.enhanced_error_handler = EnhancedErrorHandler(config)

        # Performance profiling for indexing operations
        self._metrics_collector = MetricsCollector()
        self._profiler = PerformanceProfiler(self._metrics_collector)

    async def initialize(self) -> None:
        """Initialize the indexing engine and load default indexes."""
        # Load default indexes based on configuration
        await self._load_default_indexes()

    async def _load_default_indexes(self) -> None:
        """Load default index implementations."""
        # Import and add default indexes
        try:
            from ..indexes.code_snippets_index import CodeSnippetsIndex

            self.coordinator.add_index(CodeSnippetsIndex(self.config))
        except ImportError:
            logger.warning("Code snippets index not available")

        try:
            from ..indexes.full_text_index import FullTextIndex

            self.coordinator.add_index(FullTextIndex(self.config))
        except ImportError:
            logger.warning("Full text index not available")

        try:
            from ..indexes.chunk_index import ChunkIndex

            self.coordinator.add_index(ChunkIndex(self.config))
        except ImportError:
            logger.warning("Chunk index not available")

        if self.config.enable_metadata_indexing:
            try:
                from ..indexes.vector_index import VectorIndex

                self.coordinator.add_index(VectorIndex(self.config))
            except ImportError:
                logger.warning("Vector index not available")

    async def refresh_index(
        self,
        directories: list[str] | None = None,
        branch: str | None = None,
        repo_name: str | None = None,
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
                    timeout=5.0,
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
                    timeout=5.0,
                )
                if result.returncode == 0:
                    url = result.stdout.strip()
                    repo_name = Path(url).stem.replace(".git", "")
            except Exception:
                repo_name = Path(directories[0]).name

        for directory in directories:
            yield IndexingProgressUpdate(
                progress=0.0, description=f"Starting indexing for {directory}", status="loading"
            )

            # Profile the entire indexing operation for this directory
            async with self._profiler.profile_operation(
                f"refresh_index_{directory}",
                metadata={"directory": directory, "branch": branch},
            ) as profile_id:
                # Discover files
                yield IndexingProgressUpdate(
                    progress=0.1, description=f"Discovering files in {directory}", status="indexing"
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
                        self.error_collector.add_error(e, file_path=file_path)
                        # Use enhanced error handler for recovery attempts
                        await self.enhanced_error_handler.handle_error(
                            context=str(file_path),
                            exception=e,
                            severity=ErrorSeverity.WARNING,
                            metadata={"operation": "file_discovery"},
                        )

                yield IndexingProgressUpdate(
                    progress=0.2,
                    description=f"Found {file_count} files to process",
                    status="indexing",
                )

                # Update profiler with file count
                await self._profiler.update_profile_stats(profile_id, files_processed=file_count)

                # Create tag for this indexing operation
                tag = IndexTag(
                    directory=directory, branch=branch, artifact_id="*"
                )  # Will be replaced per index

                # Read file function
                def read_file(path: str) -> str:
                    result = read_text_safely(Path(path))
                    return result if result is not None else ""

                # Refresh all indexes
                progress_offset = 0.2
                progress_scale = 0.8

                async for update in self.coordinator.refresh_all_indexes(
                    tag, current_files, read_file, repo_name
                ):
                    if self._paused:
                        yield IndexingProgressUpdate(
                            progress=update.progress, description="Indexing paused", status="paused"
                        )

                        while self._paused and not self._cancel_event.is_set():
                            await asyncio.sleep(0.1)

                        if self._cancel_event.is_set():
                            yield IndexingProgressUpdate(
                                progress=update.progress,
                                description="Indexing cancelled",
                                status="cancelled",
                            )
                            return

                    # Scale progress to account for file discovery
                    scaled_progress = progress_offset + (update.progress * progress_scale)

                    yield IndexingProgressUpdate(
                        progress=scaled_progress,
                        description=update.description,
                        status=update.status,
                        warnings=update.warnings,
                        debug_info=update.debug_info,
                    )

    async def refresh_file(
        self,
        file_path: str,
        directory: str,
        branch: str | None = None,
        repo_name: str | None = None,
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
            async with self._profiler.profile_operation(
                f"refresh_file_{Path(file_path).name}",
                metadata={"file_path": file_path, "directory": directory},
            ) as profile_id:
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
                tag = IndexTag(directory=directory, branch=branch, artifact_id="*")

                # Read file function
                def read_file(path: str) -> str:
                    result = read_text_safely(Path(path))
                    return result if result is not None else ""

                # Refresh all indexes for this file
                async for update in self.coordinator.refresh_all_indexes(
                    tag, current_files, read_file, repo_name
                ):
                    # Log progress for single file updates
                    logger.debug(f"File refresh progress: {update.description}")

                await self._profiler.update_profile_stats(profile_id, files_processed=1)

        except Exception as e:
            logger.error(f"Error refreshing file {file_path}: {e}")
            self.error_collector.add_error(e, file_path=Path(file_path))
            # Attempt recovery via enhanced error handler
            await self.enhanced_error_handler.handle_error(
                context=file_path,
                exception=e,
                severity=ErrorSeverity.ERROR,
                metadata={"operation": "refresh_file", "directory": directory},
            )

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

    async def get_index_stats(self) -> dict[str, Any]:
        """Get statistics for all indexes."""
        stats = {
            "total_indexes": len(self.coordinator.indexes),
            "index_types": [idx.artifact_id for idx in self.coordinator.indexes],
            "cache_dir": str(self.config.resolve_cache_dir()),
            "errors": [str(error) for error in self.error_collector.errors],
        }

        # Add global cache stats
        try:
            cache_stats = await self._get_cache_stats()
            stats["cache"] = cache_stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")

        # Add enhanced error and recovery report
        stats["enhanced_error_report"] = self.enhanced_error_handler.get_comprehensive_report()

        # Add profiling metrics
        stats["profiling_metrics"] = self._metrics_collector.get_all_aggregates()

        return stats

    async def _get_cache_stats(self) -> dict[str, Any]:
        """Get global cache statistics."""
        try:
            stats = await self.coordinator.global_cache.get_stats()  # type: ignore[attr-defined]
            return stats
        except Exception:
            # Fallback if global cache doesn't support stats yet
            return {
                "total_entries": 0,
                "total_size_bytes": 0,
                "hit_rate": 0.0,
            }
