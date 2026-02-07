"""
Cache cleanup and maintenance functionality.

This module provides automatic cleanup of expired cache entries and
background maintenance tasks for optimal cache performance.

Classes:
    CacheCleanup: Handles cache cleanup and maintenance operations

Features:
    - Automatic cleanup of expired entries
    - Background cleanup thread management
    - Configurable cleanup intervals
    - Thread-safe cleanup operations
    - Graceful shutdown handling
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ...utils.logging_config import get_logger

logger = get_logger()


class CacheCleanup:
    """
    Manages cache cleanup and maintenance operations.

    This class handles automatic cleanup of expired cache entries using
    a background thread and provides manual cleanup capabilities.
    """

    def __init__(
        self,
        cleanup_callback: Callable[[], int],
        cleanup_interval: float = 300,
        auto_cleanup: bool = True,  # 5 minutes
    ):
        """
        Initialize cache cleanup manager.

        Args:
            cleanup_callback: Function to call for cleanup (should return number of items cleaned)
            cleanup_interval: Interval between cleanup runs in seconds
            auto_cleanup: Whether to start automatic cleanup
        """
        self.cleanup_callback = cleanup_callback
        self.cleanup_interval = cleanup_interval
        self.auto_cleanup = auto_cleanup

        # Cleanup thread management
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_stop_event = threading.Event()

        if auto_cleanup:
            self.start_cleanup_thread()

    def start_cleanup_thread(self) -> None:
        """Start the automatic cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.warning("Cleanup thread is already running")
            return

        self._cleanup_stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True, name="CacheCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Cache cleanup thread started")

    def stop_cleanup_thread(self, timeout: float = 5.0) -> None:
        """
        Stop the automatic cleanup thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self._cleanup_thread or not self._cleanup_thread.is_alive():
            return

        self._cleanup_stop_event.set()
        self._cleanup_thread.join(timeout=timeout)

        if self._cleanup_thread.is_alive():
            logger.warning("Cleanup thread did not stop within timeout")
        else:
            logger.info("Cache cleanup thread stopped")

    def _cleanup_worker(self) -> None:
        """Background worker that performs periodic cleanup."""
        logger.debug(f"Cleanup worker started with interval {self.cleanup_interval}s")

        while not self._cleanup_stop_event.wait(self.cleanup_interval):
            try:
                start_time = time.time()
                removed_count = self.cleanup_callback()
                elapsed = time.time() - start_time

                if removed_count > 0:
                    logger.debug(f"Cleanup removed {removed_count} entries in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")

        logger.debug("Cleanup worker stopped")

    def manual_cleanup(self) -> int:
        """
        Perform manual cleanup operation.

        Returns:
            Number of items cleaned up
        """
        try:
            return self.cleanup_callback()
        except Exception as e:
            logger.error(f"Error in manual cleanup: {e}")
            return 0

    def is_running(self) -> bool:
        """Check if the cleanup thread is running."""
        return (
            self._cleanup_thread is not None
            and self._cleanup_thread.is_alive()
            and not self._cleanup_stop_event.is_set()
        )

    def get_status(self) -> dict[str, Any]:
        """
        Get cleanup status information.

        Returns:
            Dictionary with cleanup status details
        """
        return {
            "auto_cleanup_enabled": self.auto_cleanup,
            "cleanup_interval": self.cleanup_interval,
            "thread_running": self.is_running(),
            "thread_alive": self._cleanup_thread.is_alive() if self._cleanup_thread else False,
        }

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Shutdown the cleanup manager.

        Args:
            timeout: Maximum time to wait for cleanup thread to stop
        """
        self.stop_cleanup_thread(timeout)
        logger.info("Cache cleanup manager shutdown complete")
