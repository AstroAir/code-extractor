"""
Real-time file monitoring and auto-update functionality.

This module provides comprehensive file watching capabilities for automatic
index updates and cache invalidation when files change.

Classes:
    FileWatchingManager: Manages file watching and auto-update functionality

Key Features:
    - Real-time file change monitoring
    - Automatic index updates on file changes
    - Cache invalidation on file modifications
    - Configurable debouncing and batching
    - Custom change handlers

Example:
    Using file watching:
        >>> from pysearch.core.managers.file_watching import FileWatchingManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["."])
        >>> manager = FileWatchingManager(config)
        >>> manager.enable_auto_watch()
        >>> # File changes will now automatically update the search index
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import SearchConfig


class FileWatchingManager:
    """Manages file watching and auto-update functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.watch_manager: Any = None
        self._auto_watch_enabled = False
        self._indexer: Any = None

    def _ensure_watch_manager(self) -> None:
        """Lazy load the watch manager to avoid circular imports."""
        if self.watch_manager is None:
            from ...utils.file_watcher import WatchManager

            self.watch_manager = WatchManager()

    def set_indexer(self, indexer: Any) -> None:
        """Set the indexer instance for auto-updates."""
        self._indexer = indexer

    def enable_auto_watch(
        self, debounce_delay: float = 0.5, batch_size: int = 50, max_batch_delay: float = 5.0
    ) -> bool:
        """
        Enable automatic file watching for real-time index updates.

        When enabled, the search index will be automatically updated when files
        change in the configured search paths. This provides real-time search
        results without manual index refreshing.

        Args:
            debounce_delay: Delay in seconds before processing file changes
            batch_size: Maximum number of changes to batch together
            max_batch_delay: Maximum delay before processing a batch

        Returns:
            True if auto-watch was enabled successfully, False otherwise
        """
        if self._auto_watch_enabled:
            return True

        self._ensure_watch_manager()

        try:
            # Create watchers for all configured paths
            for i, path in enumerate(self.config.paths):
                watcher_name = f"path_{i}"
                if self.watch_manager:
                    success = self.watch_manager.add_watcher(
                        name=watcher_name,
                        path=path,
                        config=self.config,
                        indexer=self._indexer,
                        debounce_delay=debounce_delay,
                        batch_size=batch_size,
                        max_batch_delay=max_batch_delay,
                    )

                if not success:
                    return False

            # Start all watchers
            if self.watch_manager:
                started = self.watch_manager.start_all()
            else:
                started = 0
            if started == len(self.config.paths):
                self._auto_watch_enabled = True
                return True
            else:
                if self.watch_manager:
                    self.watch_manager.stop_all()
                return False

        except Exception:
            return False

    def disable_auto_watch(self) -> None:
        """
        Disable automatic file watching.

        Stops all file watchers and disables real-time index updates.
        The search index will need to be manually refreshed after this.
        """
        if not self._auto_watch_enabled:
            return

        try:
            if self.watch_manager:
                self.watch_manager.stop_all()
            self._auto_watch_enabled = False
        except Exception:
            pass

    def is_auto_watch_enabled(self) -> bool:
        """
        Check if automatic file watching is enabled.

        Returns:
            True if auto-watch is enabled, False otherwise
        """
        return self._auto_watch_enabled

    def get_watch_stats(self) -> dict[str, Any]:
        """
        Get file watching statistics.

        Returns:
            Dictionary with detailed watching statistics for all watchers
        """
        if not self.watch_manager:
            return {}

        try:
            return self.watch_manager.get_all_stats()  # type: ignore[no-any-return]
        except Exception:
            return {}

    def add_custom_watcher(
        self,
        name: str,
        path: Path | str,
        change_handler: Callable[[list[Any]], None],
        **kwargs: Any,
    ) -> bool:
        """
        Add a custom file watcher with a specific change handler.

        This allows for custom processing of file changes beyond the standard
        index updates. Useful for implementing custom workflows or integrations.

        Args:
            name: Unique name for the watcher
            path: Path to watch for changes
            change_handler: Function to call when files change
            **kwargs: Additional arguments for the FileWatcher

        Returns:
            True if watcher was added successfully, False otherwise
        """
        self._ensure_watch_manager()

        try:
            if self.watch_manager:
                return self.watch_manager.add_watcher(  # type: ignore[no-any-return]
                    name=name,
                    path=path,
                    config=self.config,
                    change_handler=change_handler,
                    **kwargs,
                )
            return False
        except Exception:
            return False

    def remove_watcher(self, name: str) -> bool:
        """
        Remove a file watcher by name.

        Args:
            name: Name of the watcher to remove

        Returns:
            True if watcher was removed successfully, False otherwise
        """
        if not self.watch_manager:
            return False

        try:
            return self.watch_manager.remove_watcher(name)  # type: ignore[no-any-return]
        except Exception:
            return False

    def list_watchers(self) -> list[str]:
        """
        Get list of active file watcher names.

        Returns:
            List of watcher names
        """
        if not self.watch_manager:
            return []

        try:
            return self.watch_manager.list_watchers()  # type: ignore[no-any-return]
        except Exception:
            return []

    def pause_watching(self) -> None:
        """Temporarily pause all file watchers."""
        if self.watch_manager:
            try:
                self.watch_manager.pause_all()
            except Exception:
                pass

    def resume_watching(self) -> None:
        """Resume all paused file watchers."""
        if self.watch_manager:
            try:
                self.watch_manager.resume_all()
            except Exception:
                pass

    def get_watcher_status(self, name: str) -> dict[str, Any]:
        """
        Get status information for a specific watcher.

        Args:
            name: Name of the watcher

        Returns:
            Dictionary with watcher status information
        """
        if not self.watch_manager:
            return {}

        try:
            return self.watch_manager.get_watcher_status(name)  # type: ignore[no-any-return]
        except Exception:
            return {}

    def set_watch_filters(
        self, include_patterns: list[str] | None = None, exclude_patterns: list[str] | None = None
    ) -> None:
        """
        Set file patterns to include or exclude from watching.

        Args:
            include_patterns: Patterns for files to include in watching
            exclude_patterns: Patterns for files to exclude from watching
        """
        if self.watch_manager:
            try:
                self.watch_manager.set_filters(include_patterns, exclude_patterns)
            except Exception:
                pass

    def force_rescan(self) -> bool:
        """
        Force a rescan of all watched directories.

        Returns:
            True if rescan was successful, False otherwise
        """
        if not self.watch_manager or not self._indexer:
            return False

        try:
            # Trigger a manual scan through the indexer
            self._indexer.scan()
            return True
        except Exception:
            return False

    def get_watch_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for file watching operations.

        Returns:
            Dictionary with performance metrics
        """
        if not self.watch_manager:
            return {}

        try:
            stats = self.watch_manager.get_all_stats()

            # Calculate aggregate metrics
            total_events = sum(
                watcher_stats.get("events_processed", 0) for watcher_stats in stats.values()
            )

            total_errors = sum(watcher_stats.get("errors", 0) for watcher_stats in stats.values())

            avg_processing_time = 0.0
            if stats:
                processing_times = [
                    watcher_stats.get("avg_processing_time", 0.0)
                    for watcher_stats in stats.values()
                ]
                avg_processing_time = sum(processing_times) / len(processing_times)

            return {
                "total_watchers": len(stats),
                "total_events_processed": total_events,
                "total_errors": total_errors,
                "average_processing_time": avg_processing_time,
                "error_rate": total_errors / total_events if total_events > 0 else 0.0,
                "individual_watchers": stats,
            }
        except Exception:
            return {}
