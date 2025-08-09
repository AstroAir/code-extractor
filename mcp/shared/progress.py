#!/usr/bin/env python3
"""
MCP Progress Reporting for PySearch

This module implements progress reporting functionality for long-running
search operations using FastMCP progress reporting capabilities.

Features:
- Real-time progress updates for search operations
- Cancellation support for long-running tasks
- Progress tracking for file analysis and indexing
- Batch operation progress reporting
- Performance monitoring during operations
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import MCP server
from ..servers.mcp_server import PySearchMCPServer


class ProgressStatus(Enum):
    """Status of a progress-tracked operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    """Progress update information."""

    operation_id: str
    status: ProgressStatus
    progress: float  # 0.0 to 1.0
    current_step: str
    total_steps: int
    completed_steps: int
    elapsed_time: float
    estimated_remaining: float | None = None
    details: dict[str, Any] | None = None


class ProgressTracker:
    """
    Tracks progress for long-running operations.

    Provides real-time progress updates and cancellation support
    for search operations, file analysis, and batch processing.
    """

    def __init__(self) -> None:
        self.active_operations: dict[str, ProgressUpdate] = {}
        self.operation_callbacks: dict[str, list[Callable[[ProgressUpdate], None]]] = {}
        self.cancellation_flags: dict[str, bool] = {}

    def start_operation(
        self, operation_id: str, total_steps: int, description: str = "Processing"
    ) -> None:
        """Start tracking a new operation."""
        self.active_operations[operation_id] = ProgressUpdate(
            operation_id=operation_id,
            status=ProgressStatus.RUNNING,
            progress=0.0,
            current_step=description,
            total_steps=total_steps,
            completed_steps=0,
            elapsed_time=0.0,
        )
        self.cancellation_flags[operation_id] = False

    def update_progress(
        self,
        operation_id: str,
        completed_steps: int,
        current_step: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Update progress for an operation."""
        if operation_id not in self.active_operations:
            return

        operation = self.active_operations[operation_id]
        operation.completed_steps = completed_steps
        operation.current_step = current_step
        operation.progress = completed_steps / operation.total_steps
        operation.details = details or {}

        # Calculate elapsed time and estimate remaining
        start_time = getattr(operation, "_start_time", time.time())
        operation.elapsed_time = time.time() - start_time

        if operation.progress > 0:
            total_estimated = operation.elapsed_time / operation.progress
            operation.estimated_remaining = total_estimated - operation.elapsed_time

        # Notify callbacks
        self._notify_callbacks(operation_id, operation)

    def complete_operation(self, operation_id: str, success: bool = True) -> None:
        """Mark an operation as completed."""
        if operation_id not in self.active_operations:
            return

        operation = self.active_operations[operation_id]
        operation.status = ProgressStatus.COMPLETED if success else ProgressStatus.FAILED
        operation.progress = 1.0
        operation.completed_steps = operation.total_steps

        self._notify_callbacks(operation_id, operation)

    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an operation."""
        if operation_id not in self.active_operations:
            return False

        self.cancellation_flags[operation_id] = True
        operation = self.active_operations[operation_id]
        operation.status = ProgressStatus.CANCELLED

        self._notify_callbacks(operation_id, operation)
        return True

    def is_cancelled(self, operation_id: str) -> bool:
        """Check if an operation has been cancelled."""
        return self.cancellation_flags.get(operation_id, False)

    def get_operation_status(self, operation_id: str) -> ProgressUpdate | None:
        """Get current status of an operation."""
        return self.active_operations.get(operation_id)

    def add_callback(self, operation_id: str, callback: Callable) -> None:
        """Add a progress callback for an operation."""
        if operation_id not in self.operation_callbacks:
            self.operation_callbacks[operation_id] = []
        self.operation_callbacks[operation_id].append(callback)

    def _notify_callbacks(self, operation_id: str, update: ProgressUpdate) -> None:
        """Notify all callbacks for an operation."""
        callbacks = self.operation_callbacks.get(operation_id, [])
        for callback in callbacks:
            try:
                callback(update)
            except Exception:
                # Ignore callback errors
                pass

    def cleanup_completed(self, max_age_seconds: int = 3600) -> None:
        """Clean up completed operations older than max_age_seconds."""
        current_time = time.time()
        to_remove = []

        for operation_id, operation in self.active_operations.items():
            if operation.status in [
                ProgressStatus.COMPLETED,
                ProgressStatus.FAILED,
                ProgressStatus.CANCELLED,
            ]:
                if current_time - operation.elapsed_time > max_age_seconds:
                    to_remove.append(operation_id)

        for operation_id in to_remove:
            del self.active_operations[operation_id]
            if operation_id in self.operation_callbacks:
                del self.operation_callbacks[operation_id]
            if operation_id in self.cancellation_flags:
                del self.cancellation_flags[operation_id]


class ProgressAwareSearchServer(PySearchMCPServer):
    """
    PySearch MCP Server with progress reporting capabilities.

    Extends the main server to provide progress updates for
    long-running operations like batch searches and file analysis.
    """

    def __init__(self, name: str = "Progress-Aware PySearch MCP Server") -> None:
        super().__init__(name)
        self.progress_tracker = ProgressTracker()

    def _get_progress_update(self, operation_id: str) -> ProgressUpdate:
        """Get progress update, ensuring we always return a valid ProgressUpdate."""
        status = self.progress_tracker.get_operation_status(operation_id)
        if status is None:
            # Return a default status if none exists
            return ProgressUpdate(
                operation_id=operation_id,
                status=ProgressStatus.RUNNING,
                progress=0.0,
                current_step="Unknown",
                total_steps=1,
                completed_steps=0,
                elapsed_time=0.0,
            )
        return status

    async def search_with_progress(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        use_regex: bool = False,
        progress_callback: Callable | None = None,
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """
        Perform search with progress reporting.

        Args:
            pattern: Search pattern
            paths: Optional list of paths to search
            context: Number of context lines around matches
            use_regex: Whether to use regex search
            progress_callback: Optional callback for progress updates

        Yields:
            ProgressUpdate objects with current progress
        """
        operation_id = f"search_{int(time.time() * 1000)}"

        try:
            # Start progress tracking
            self.progress_tracker.start_operation(
                operation_id,
                total_steps=5,  # Simplified: init, scan, search, process, complete
                description="Initializing search",
            )

            if progress_callback:
                self.progress_tracker.add_callback(operation_id, progress_callback)

            # Step 1: Initialize
            yield self._get_progress_update(operation_id)
            await asyncio.sleep(0.1)  # Allow for cancellation check

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Step 2: Scan files
            self.progress_tracker.update_progress(
                operation_id, 1, "Scanning files", {"phase": "file_discovery"}
            )
            yield self._get_progress_update(operation_id)
            await asyncio.sleep(0.1)

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Step 3: Execute search
            self.progress_tracker.update_progress(
                operation_id, 2, "Executing search", {"phase": "pattern_matching"}
            )
            yield self._get_progress_update(operation_id)

            # Perform actual search
            result = await self.search_text(pattern, paths, context)

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Step 4: Process results
            self.progress_tracker.update_progress(
                operation_id,
                3,
                "Processing results",
                {"phase": "result_processing", "matches_found": result.total_matches},
            )
            yield self._get_progress_update(operation_id)
            await asyncio.sleep(0.1)

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Step 5: Complete
            self.progress_tracker.update_progress(
                operation_id, 4, "Finalizing", {"phase": "completion"}
            )
            yield self._get_progress_update(operation_id)

            # Mark as completed
            self.progress_tracker.complete_operation(operation_id, True)
            yield self._get_progress_update(operation_id)

        except Exception as e:
            self.progress_tracker.complete_operation(operation_id, False)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {"error": str(e)}
            yield final_update

    async def batch_file_analysis_with_progress(
        self, file_paths: list[str], progress_callback: Callable | None = None
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """
        Perform batch file analysis with progress reporting.

        Args:
            file_paths: List of file paths to analyze
            progress_callback: Optional callback for progress updates

        Yields:
            ProgressUpdate objects with current progress
        """
        operation_id = f"batch_analysis_{int(time.time() * 1000)}"

        try:
            # Start progress tracking
            self.progress_tracker.start_operation(
                operation_id,
                total_steps=len(file_paths),
                description="Starting batch file analysis",
            )

            if progress_callback:
                self.progress_tracker.add_callback(operation_id, progress_callback)

            # Initial progress update
            yield self._get_progress_update(operation_id)

            # Process each file
            for i, file_path in enumerate(file_paths):
                if self.progress_tracker.is_cancelled(operation_id):
                    break

                try:
                    # Analyze the file
                    analysis = await self.analyze_file_content(file_path, True, True)

                    # Update progress
                    self.progress_tracker.update_progress(
                        operation_id,
                        i + 1,
                        f"Analyzed {file_path}",
                        {
                            "current_file": file_path,
                            "complexity_score": analysis.complexity_score,
                            "quality_score": analysis.code_quality_score,
                        },
                    )

                    yield self._get_progress_update(operation_id)

                    # Small delay to allow for cancellation
                    await asyncio.sleep(0.01)

                except Exception as e:
                    # Continue with next file on error
                    self.progress_tracker.update_progress(
                        operation_id,
                        i + 1,
                        f"Error analyzing {file_path}",
                        {"current_file": file_path, "error": str(e)},
                    )
                    yield self._get_progress_update(operation_id)

            # Mark as completed
            success = not self.progress_tracker.is_cancelled(operation_id)
            self.progress_tracker.complete_operation(operation_id, success)
            yield self._get_progress_update(operation_id)

        except Exception as e:
            self.progress_tracker.complete_operation(operation_id, False)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {"error": str(e)}
            yield final_update

    def get_active_operations(self) -> list[ProgressUpdate]:
        """Get list of all active operations."""
        return list(self.progress_tracker.active_operations.values())

    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation."""
        return self.progress_tracker.cancel_operation(operation_id)

    def cleanup_old_operations(self) -> dict[str, str]:
        """Clean up old completed operations."""
        before_count = len(self.progress_tracker.active_operations)
        self.progress_tracker.cleanup_completed()
        after_count = len(self.progress_tracker.active_operations)

        return {
            "status": "success",
            "message": f"Cleaned up {before_count - after_count} old operations",
        }


def create_progress_aware_server() -> ProgressAwareSearchServer:
    """Create and configure the progress-aware MCP server instance."""
    return ProgressAwareSearchServer()
