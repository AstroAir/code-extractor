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
        self._cache_invalidation_callback: Callable[[list[Any]], None] | None = None

    def _ensure_watch_manager(self) -> None:
        """Lazy load the watch manager to avoid circular imports."""
        if self.watch_manager is None:
            from ...utils.file_watcher import WatchManager

            self.watch_manager = WatchManager()

    def set_indexer(self, indexer: Any) -> None:
        """Set the indexer instance for auto-updates."""
        self._indexer = indexer

    def set_cache_invalidation_callback(self, callback: Callable[[list[Any]], None]) -> None:
        """Set the callback to invoke when files change, for cache invalidation.

        Args:
            callback: Function accepting a list of changed file Paths.
        """
        self._cache_invalidation_callback = callback

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

        if not self.watch_manager:
            return False

        try:
            # Create watchers for all configured paths
            for i, path in enumerate(self.config.paths):
                watcher_name = f"path_{i}"
                success = self.watch_manager.add_watcher(
                    name=watcher_name,
                    path=path,
                    config=self.config,
                    indexer=self._indexer,
                    debounce_delay=debounce_delay,
                    batch_size=batch_size,
                    max_batch_delay=max_batch_delay,
                    cache_invalidation_callback=self._cache_invalidation_callback,
                )
                if not success:
                    # Rollback any watchers already added
                    self.watch_manager.stop_all()
                    return False

            # Start all watchers
            started = self.watch_manager.start_all()
            if started == len(self.config.paths):
                self._auto_watch_enabled = True
                return True
            else:
                self.watch_manager.stop_all()
                return False

        except Exception:
            if self.watch_manager:
                self.watch_manager.stop_all()
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
            return self.watch_manager.get_all_stats()
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
                return self.watch_manager.add_watcher(
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
            return self.watch_manager.remove_watcher(name)
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
            return self.watch_manager.list_watchers()
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
            return self.watch_manager.get_watcher_status(name)
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

    def enable_workspace_watch(
        self,
        workspace_config: Any,
        debounce_delay: float = 0.5,
        batch_size: int = 50,
        max_batch_delay: float = 5.0,
    ) -> dict[str, bool]:
        """
        Enable file watching for all repositories in a workspace.

        Creates a watcher for each enabled repository in the workspace,
        using its configured include/exclude patterns.

        Args:
            workspace_config: WorkspaceConfig instance
            debounce_delay: Delay before processing changes
            batch_size: Max changes to batch together
            max_batch_delay: Max delay before processing a batch

        Returns:
            Dictionary mapping repository names to watch success status
        """
        self._ensure_watch_manager()
        if not self.watch_manager:
            return {}

        from pathlib import Path as _Path

        results: dict[str, bool] = {}

        for repo_cfg in workspace_config.get_enabled_repositories():
            repo_path = _Path(repo_cfg.path)
            if not repo_path.is_absolute():
                repo_path = _Path(workspace_config.root_path) / repo_path

            if not repo_path.exists():
                results[repo_cfg.name] = False
                continue

            # Build per-repo config for watching
            include = repo_cfg.include or workspace_config.include
            exclude = repo_cfg.exclude or workspace_config.exclude

            from ..config import SearchConfig as _SC

            watch_cfg = _SC(
                paths=[str(repo_path)],
                include=include,
                exclude=exclude,
                context=workspace_config.context,
                parallel=workspace_config.parallel,
                workers=workspace_config.workers,
                follow_symlinks=workspace_config.follow_symlinks,
            )

            watcher_name = f"ws_{repo_cfg.name}"
            try:
                success = self.watch_manager.add_watcher(
                    name=watcher_name,
                    path=str(repo_path),
                    config=watch_cfg,
                    indexer=self._indexer,
                    debounce_delay=debounce_delay,
                    batch_size=batch_size,
                    max_batch_delay=max_batch_delay,
                    cache_invalidation_callback=self._cache_invalidation_callback,
                )
                results[repo_cfg.name] = bool(success)
            except Exception:
                results[repo_cfg.name] = False

        # Start all newly added watchers
        if any(results.values()):
            try:
                self.watch_manager.start_all()
                self._auto_watch_enabled = True
            except Exception:
                pass

        return results

    def disable_workspace_watch(self, workspace_config: Any) -> None:
        """
        Disable file watching for all workspace repositories.

        Args:
            workspace_config: WorkspaceConfig instance
        """
        if not self.watch_manager:
            return

        for repo_cfg in workspace_config.repositories:
            watcher_name = f"ws_{repo_cfg.name}"
            try:
                self.watch_manager.remove_watcher(watcher_name)
            except Exception:
                pass

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
