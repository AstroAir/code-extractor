"""
Real-time file watching module for pysearch.

This module provides comprehensive file system monitoring capabilities including:
- Real-time file change detection using platform-specific APIs
- Incremental search index updates
- Batch change processing for performance
- Configurable watch patterns and exclusions
- Event debouncing and throttling
- Cross-platform compatibility

Classes:
    FileEvent: Represents a file system event
    FileWatcher: Main file watching engine
    WatchManager: Manages multiple watchers
    ChangeProcessor: Processes file changes for search updates

Features:
    - Uses watchdog library for cross-platform file monitoring
    - Intelligent change batching to avoid excessive updates
    - Configurable debouncing to handle rapid file changes
    - Integration with PySearch indexer for automatic updates
    - Support for include/exclude patterns
    - Memory-efficient event handling

Example:
    Basic file watching:
        >>> from pysearch.file_watcher import FileWatcher
        >>> watcher = FileWatcher("/path/to/project")
        >>> watcher.start()
        >>> # File changes are automatically detected and processed
        >>> watcher.stop()

    Advanced watching with custom handler:
        >>> def on_change(events):
        ...     print(f"Files changed: {[e.path for e in events]}")
        >>>
        >>> watcher = FileWatcher("/path/to/project", change_handler=on_change)
        >>> watcher.start()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    # Fallback classes for when watchdog is not available

    class _FallbackFileSystemEvent:
        def __init__(self, src_path: str):
            self.src_path = src_path
            self.is_directory = False

    class _FallbackFileSystemEventHandler:
        pass

    class _FallbackObserver:
        def __init__(self) -> None:
            pass

        def schedule(self, handler: Any, path: str, recursive: bool = True) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def join(self) -> None:
            pass

    # Assign fallback classes to the expected names
    FileSystemEvent = _FallbackFileSystemEvent  # type: ignore
    FileSystemEventHandler = _FallbackFileSystemEventHandler  # type: ignore
    Observer = _FallbackObserver  # type: ignore

from ..core.config import SearchConfig
from ..indexing.indexer import Indexer
from .logging_config import get_logger
from .utils import matches_patterns


class EventType(Enum):
    """Types of file system events."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """Represents a file system event with metadata."""

    path: Path
    event_type: EventType
    timestamp: float
    is_directory: bool = False
    old_path: Path | None = None  # For move events
    metadata: dict[str, Any] = field(default_factory=dict)


class ChangeProcessor:
    """
    Processes file changes for search index updates.

    Handles batching, debouncing, and intelligent processing of file changes
    to maintain search index consistency while minimizing performance impact.
    """

    def __init__(
        self,
        indexer: Indexer,
        batch_size: int = 50,
        debounce_delay: float = 0.5,
        max_batch_delay: float = 5.0
    ):
        self.indexer = indexer
        self.batch_size = batch_size
        self.debounce_delay = debounce_delay
        self.max_batch_delay = max_batch_delay
        self.logger = get_logger()

        # Event processing
        self._pending_events: dict[Path, FileEvent] = {}
        self._processing_lock = threading.RLock()
        self._last_batch_time = time.time()
        self._debounce_timer: threading.Timer | None = None

        # Statistics
        self.events_processed = 0
        self.batches_processed = 0
        self.last_processing_time = 0.0

    def process_event(self, event: FileEvent) -> None:
        """
        Process a single file system event.

        Args:
            event: File system event to process
        """
        with self._processing_lock:
            # Update pending events (latest event wins for each path)
            self._pending_events[event.path] = event

            # Cancel existing debounce timer
            if self._debounce_timer:
                self._debounce_timer.cancel()

            # Check if we should process immediately
            should_process_now = (
                len(self._pending_events) >= self.batch_size or
                time.time() - self._last_batch_time >= self.max_batch_delay
            )

            if should_process_now:
                self._process_pending_events()
            else:
                # Set up debounce timer
                self._debounce_timer = threading.Timer(
                    self.debounce_delay,
                    self._process_pending_events
                )
                self._debounce_timer.start()

    def _process_pending_events(self) -> None:
        """Process all pending events in a batch."""
        with self._processing_lock:
            if not self._pending_events:
                return

            start_time = time.time()
            events = list(self._pending_events.values())
            self._pending_events.clear()
            self._last_batch_time = start_time

            # Cancel debounce timer if active
            if self._debounce_timer:
                self._debounce_timer.cancel()
                self._debounce_timer = None

        # Process events outside the lock
        self._process_event_batch(events)

        # Update statistics
        processing_time = time.time() - start_time
        self.events_processed += len(events)
        self.batches_processed += 1
        self.last_processing_time = processing_time

        self.logger.debug(
            f"Processed {len(events)} file events in {processing_time:.3f}s"
        )

    def _process_event_batch(self, events: list[FileEvent]) -> None:
        """Process a batch of file events."""
        try:
            # Group events by type for efficient processing
            created_files = []
            modified_files = []
            deleted_files = []

            for event in events:
                if event.is_directory:
                    continue  # Skip directory events

                if event.event_type == EventType.CREATED:
                    created_files.append(event.path)
                elif event.event_type == EventType.MODIFIED:
                    modified_files.append(event.path)
                elif event.event_type == EventType.DELETED:
                    deleted_files.append(event.path)
                elif event.event_type == EventType.MOVED:
                    # Treat as delete old + create new
                    if event.old_path:
                        deleted_files.append(event.old_path)
                    created_files.append(event.path)

            # Update indexer
            if created_files or modified_files or deleted_files:
                # Trigger a rescan to update the index
                # The indexer's scan() method will detect changes and removals automatically
                try:
                    self.indexer.scan()
                except Exception as e:
                    self.logger.error(f"Error updating indexer: {e}")

            self.logger.info(
                f"Index updated: {len(created_files)} created, "
                f"{len(modified_files)} modified, {len(deleted_files)} deleted"
            )

        except Exception as e:
            self.logger.error(f"Error processing file events: {e}")

    def flush_pending(self) -> None:
        """Force processing of all pending events."""
        self._process_pending_events()

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "events_processed": self.events_processed,
            "batches_processed": self.batches_processed,
            "pending_events": len(self._pending_events),
            "last_processing_time": self.last_processing_time,
            "average_batch_size": (
                self.events_processed / self.batches_processed
                if self.batches_processed > 0 else 0
            )
        }


class PySearchEventHandler(FileSystemEventHandler):
    """
    File system event handler for PySearch.

    Filters and processes file system events according to PySearch configuration.
    """

    def __init__(
        self,
        config: SearchConfig,
        change_processor: ChangeProcessor,
        custom_handler: Callable[[list[FileEvent]], None] | None = None
    ):
        super().__init__()
        self.config = config
        self.change_processor = change_processor
        self.custom_handler = custom_handler
        self.logger = get_logger()

        # Event filtering
        self._recent_events: deque[tuple[str, float]] = deque(maxlen=1000)
        self._duplicate_threshold = 0.1  # seconds

    def _should_process_path(self, path: Path) -> bool:
        """Check if a path should be processed based on configuration."""
        # Check include patterns
        if self.config.include:
            if not matches_patterns(path, self.config.include):
                return False

        # Check exclude patterns
        if self.config.exclude:
            if matches_patterns(path, self.config.exclude):
                return False

        return True

    def _is_duplicate_event(self, path: str, timestamp: float) -> bool:
        """Check if this is a duplicate event that should be ignored."""
        # Remove old events
        cutoff = timestamp - self._duplicate_threshold
        while self._recent_events and self._recent_events[0][1] < cutoff:
            self._recent_events.popleft()

        # Check for duplicates
        for event_path, event_time in self._recent_events:
            if event_path == path and abs(event_time - timestamp) < self._duplicate_threshold:
                return True

        # Add this event
        self._recent_events.append((path, timestamp))
        return False

    def _create_file_event(
        self,
        src_path: str,
        event_type: EventType,
        is_directory: bool = False,
        dest_path: str | None = None
    ) -> FileEvent | None:
        """Create a FileEvent from file system event data."""
        path = Path(src_path)
        timestamp = time.time()

        # Check for duplicates
        if self._is_duplicate_event(src_path, timestamp):
            return None

        # Check if path should be processed
        if not self._should_process_path(path):
            return None

        return FileEvent(
            path=path,
            event_type=event_type,
            timestamp=timestamp,
            is_directory=is_directory,
            old_path=Path(dest_path) if dest_path else None
        )

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file/directory creation events."""
        file_event = self._create_file_event(
            str(event.src_path),
            EventType.CREATED,
            event.is_directory
        )
        if file_event:
            self._process_event(file_event)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file/directory modification events."""
        file_event = self._create_file_event(
            str(event.src_path),
            EventType.MODIFIED,
            event.is_directory
        )
        if file_event:
            self._process_event(file_event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file/directory deletion events."""
        file_event = self._create_file_event(
            str(event.src_path),
            EventType.DELETED,
            event.is_directory
        )
        if file_event:
            self._process_event(file_event)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file/directory move events."""
        # Handle move as delete + create
        if hasattr(event, 'dest_path'):
            file_event = self._create_file_event(
                str(event.dest_path),
                EventType.MOVED,
                event.is_directory,
                str(event.src_path)
            )
            if file_event:
                self._process_event(file_event)

    def _process_event(self, event: FileEvent) -> None:
        """Process a file event."""
        try:
            # Process through change processor
            self.change_processor.process_event(event)

            # Call custom handler if provided
            if self.custom_handler:
                self.custom_handler([event])

        except Exception as e:
            self.logger.error(f"Error processing file event {event.path}: {e}")


class FileWatcher:
    """
    Main file watching engine for real-time file system monitoring.

    Provides comprehensive file watching capabilities with intelligent
    change processing and integration with PySearch indexing.
    """

    def __init__(
        self,
        path: Path | str,
        config: SearchConfig | None = None,
        indexer: Indexer | None = None,
        change_handler: Callable[[list[FileEvent]], None] | None = None,
        recursive: bool = True,
        **processor_kwargs: Any
    ):
        """
        Initialize file watcher.

        Args:
            path: Path to watch for changes
            config: Search configuration (uses default if None)
            indexer: Indexer instance for automatic updates
            change_handler: Custom handler for file change events
            recursive: Whether to watch subdirectories recursively
            **processor_kwargs: Additional arguments for ChangeProcessor
        """
        self.path = Path(path)
        self.config = config or SearchConfig()
        self.recursive = recursive
        self.logger = get_logger()

        # Check if watchdog is available
        if not WATCHDOG_AVAILABLE:
            self.logger.warning(
                "Watchdog library not available. File watching will be disabled. "
                "Install with: pip install watchdog"
            )
            self._observer = None
            self._change_processor = None
            self._event_handler = None
            return

        # Initialize components
        if indexer is None:
            indexer = Indexer(self.config)

        self._change_processor = ChangeProcessor(indexer, **processor_kwargs)
        self._event_handler = PySearchEventHandler(
            self.config,
            self._change_processor,
            change_handler
        )
        self._observer = Observer()

        # State management
        self._is_watching = False
        self._watch_handle: Any = None  # ObservedWatch when watching
        self._start_time: float | None = None

    @property
    def is_available(self) -> bool:
        """Check if file watching is available."""
        return WATCHDOG_AVAILABLE and self._observer is not None

    @property
    def is_watching(self) -> bool:
        """Check if currently watching for file changes."""
        return self._is_watching

    def start(self) -> bool:
        """
        Start watching for file changes.

        Returns:
            True if watching started successfully, False otherwise
        """
        if not self.is_available:
            self.logger.warning("File watching not available")
            return False

        if self._is_watching:
            self.logger.warning("File watcher already running")
            return True

        try:
            # Check if observer is available
            if self._observer is None or self._event_handler is None:
                self.logger.error("Observer or event handler not initialized")
                return False

            # Schedule the watch
            self._watch_handle = self._observer.schedule(
                self._event_handler,
                str(self.path),
                recursive=self.recursive
            )

            # Start the observer
            self._observer.start()
            self._is_watching = True
            self._start_time = time.time()

            self.logger.info(
                f"Started watching: {self.path} (recursive={self.recursive})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start file watcher: {e}")
            return False

    def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._is_watching or not self._observer:
            return

        try:
            # Flush any pending changes
            if self._change_processor:
                self._change_processor.flush_pending()

            # Stop the observer
            self._observer.stop()
            self._observer.join(timeout=5.0)  # Wait up to 5 seconds

            self._is_watching = False
            self.logger.info("Stopped file watcher")

        except Exception as e:
            self.logger.error(f"Error stopping file watcher: {e}")

    def restart(self) -> bool:
        """
        Restart the file watcher.

        Returns:
            True if restart was successful, False otherwise
        """
        self.stop()
        return self.start()

    def get_stats(self) -> dict[str, Any]:
        """
        Get file watcher statistics.

        Returns:
            Dictionary with watcher statistics
        """
        stats = {
            "is_available": self.is_available,
            "is_watching": self.is_watching,
            "path": str(self.path),
            "recursive": self.recursive,
            "uptime": 0.0
        }

        if self._start_time and self._is_watching:
            stats["uptime"] = time.time() - self._start_time

        if self._change_processor:
            stats.update(self._change_processor.get_stats())

        return stats

    def __enter__(self) -> FileWatcher:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


class WatchManager:
    """
    Manages multiple file watchers for complex projects.

    Allows watching multiple directories with different configurations
    and provides centralized management of all watchers.
    """

    def __init__(self) -> None:
        self.watchers: dict[str, FileWatcher] = {}
        self.logger = get_logger()

    def add_watcher(
        self,
        name: str,
        path: Path | str,
        **kwargs: Any
    ) -> bool:
        """
        Add a new file watcher.

        Args:
            name: Unique name for the watcher
            path: Path to watch
            **kwargs: Arguments passed to FileWatcher

        Returns:
            True if watcher was added successfully, False otherwise
        """
        if name in self.watchers:
            self.logger.warning(f"Watcher '{name}' already exists")
            return False

        try:
            watcher = FileWatcher(path, **kwargs)
            self.watchers[name] = watcher
            self.logger.info(f"Added watcher '{name}' for path: {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add watcher '{name}': {e}")
            return False

    def remove_watcher(self, name: str) -> bool:
        """
        Remove a file watcher.

        Args:
            name: Name of the watcher to remove

        Returns:
            True if watcher was removed successfully, False otherwise
        """
        if name not in self.watchers:
            self.logger.warning(f"Watcher '{name}' not found")
            return False

        try:
            watcher = self.watchers[name]
            watcher.stop()
            del self.watchers[name]
            self.logger.info(f"Removed watcher '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to remove watcher '{name}': {e}")
            return False

    def start_all(self) -> int:
        """
        Start all watchers.

        Returns:
            Number of watchers started successfully
        """
        started = 0
        for name, watcher in self.watchers.items():
            if watcher.start():
                started += 1
            else:
                self.logger.warning(f"Failed to start watcher '{name}'")

        self.logger.info(f"Started {started}/{len(self.watchers)} watchers")
        return started

    def stop_all(self) -> None:
        """Stop all watchers."""
        for name, watcher in self.watchers.items():
            try:
                watcher.stop()
            except Exception as e:
                self.logger.error(f"Error stopping watcher '{name}': {e}")

        self.logger.info("Stopped all watchers")

    def get_watcher(self, name: str) -> FileWatcher | None:
        """Get a watcher by name."""
        return self.watchers.get(name)

    def list_watchers(self) -> list[str]:
        """Get list of watcher names."""
        return list(self.watchers.keys())

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all watchers."""
        return {name: watcher.get_stats() for name, watcher in self.watchers.items()}

    def __enter__(self) -> WatchManager:
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop_all()
