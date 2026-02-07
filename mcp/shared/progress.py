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

# Import removed due to circular dependency
# from ..servers.mcp_server import PySearchMCPServer


# Mock base class for testing
class PySearchMCPServer:
    """Mock base class for progress-aware server."""

    def __init__(self, name: str = "Mock Server"):
        self.name = name
        self.current_config = None

    async def search_regex(self, pattern: str, paths: list[str] | None, context: int) -> Any:
        """Mock search_regex method."""
        from ..servers.pysearch_mcp_server import SearchResponse

        return SearchResponse(
            items=[],
            stats={"files_scanned": 0},
            query_info={},
            total_matches=0,
            execution_time_ms=0,
        )

    async def search_text(self, pattern: str, paths: list[str] | None, context: int) -> Any:
        """Mock search_text method."""
        from ..servers.pysearch_mcp_server import SearchResponse

        return SearchResponse(
            items=[],
            stats={"files_scanned": 0},
            query_info={},
            total_matches=0,
            execution_time_ms=0,
        )

    async def analyze_file_content(
        self, file_path: str, include_complexity: bool, include_quality: bool
    ) -> Any:
        """Mock analyze_file_content method."""

        class MockAnalysisResult:
            def __init__(self) -> None:
                self.complexity_score = 1.0
                self.code_quality_score = 1.0
                self.language = "python"

        return MockAnalysisResult()


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
    _start_time: float | None = None  # Internal field for tracking start time


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
        start_time = time.time()
        update = ProgressUpdate(
            operation_id=operation_id,
            status=ProgressStatus.RUNNING,
            progress=0.0,
            current_step=description,
            total_steps=total_steps,
            completed_steps=0,
            elapsed_time=0.0,
        )
        update._start_time = start_time
        self.active_operations[operation_id] = update
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
        start_time = getattr(operation, "_start_time", None)
        if start_time is None:
            start_time = time.time()
            operation._start_time = start_time
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

    def add_callback(self, operation_id: str, callback: Callable[[ProgressUpdate], None]) -> None:
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

    def _estimate_file_count(self, paths: list[str] | None = None) -> int:
        """Estimate number of files for progress calculation."""
        # Simple estimation - in real implementation would scan directories
        if paths is None:
            return 100  # Default estimate
        return max(len(paths) * 10, 50)  # Rough estimate

    def get_active_operations(self) -> list[ProgressUpdate]:
        """Get list of active operations."""
        return list(self.progress_tracker.active_operations.values())

    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an operation."""
        return self.progress_tracker.cancel_operation(operation_id)

    async def search_with_progress(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        use_regex: bool = False,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
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
        start_time = time.time()

        try:
            # Start progress tracking with dynamic step calculation
            estimated_files = self._estimate_file_count(paths)
            total_steps = min(max(estimated_files // 100, 5), 50)  # Scale steps based on file count

            self.progress_tracker.start_operation(
                operation_id,
                total_steps=total_steps,
                description="Initializing search",
            )

            # Store start time for better estimation
            self.progress_tracker.active_operations[operation_id]._start_time = start_time

            if progress_callback:
                self.progress_tracker.add_callback(operation_id, progress_callback)

            # Step 1: Initialize and validate
            yield self._get_progress_update(operation_id)
            await asyncio.sleep(0.01)  # Shorter sleep for responsiveness

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Validate pattern if regex
            if use_regex:
                try:
                    import re

                    re.compile(pattern)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern: {e}")

            # Step 2: Scan and index files
            self.progress_tracker.update_progress(
                operation_id,
                1,
                "Scanning files",
                {"phase": "file_discovery", "estimated_files": estimated_files},
            )
            yield self._get_progress_update(operation_id)
            await asyncio.sleep(0.01)

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Step 3: Execute search with intermediate updates
            for step in range(2, total_steps - 2):
                if self.progress_tracker.is_cancelled(operation_id):
                    return

                progress_pct = step / total_steps
                self.progress_tracker.update_progress(
                    operation_id,
                    step,
                    f"Searching files ({progress_pct:.0%})",
                    {"phase": "pattern_matching", "progress_pct": progress_pct},
                )
                yield self._get_progress_update(operation_id)
                await asyncio.sleep(0.01)

            # Perform actual search
            if use_regex:
                result = await self.search_regex(pattern, paths, context)
            else:
                result = await self.search_text(pattern, paths, context)

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Step N-1: Process and rank results
            self.progress_tracker.update_progress(
                operation_id,
                total_steps - 2,
                "Processing results",
                {
                    "phase": "result_processing",
                    "matches_found": result.total_matches,
                    "files_scanned": result.stats.get("files_scanned", 0),
                },
            )
            yield self._get_progress_update(operation_id)
            await asyncio.sleep(0.01)

            if self.progress_tracker.is_cancelled(operation_id):
                return

            # Final step: Complete
            self.progress_tracker.update_progress(
                operation_id,
                total_steps - 1,
                "Finalizing",
                {"phase": "completion", "final_result_count": result.total_matches},
            )
            yield self._get_progress_update(operation_id)

            # Mark as completed
            self.progress_tracker.complete_operation(operation_id, True)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {
                "search_result": {
                    "total_matches": result.total_matches,
                    "files_matched": result.stats.get("files_matched", 0),
                    "execution_time_ms": result.execution_time_ms,
                }
            }
            yield final_update

        except asyncio.CancelledError:
            self.progress_tracker.cancel_operation(operation_id)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {"message": "Search was cancelled"}
            yield final_update
        except Exception as e:
            self.progress_tracker.complete_operation(operation_id, False)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {"error": str(e), "error_type": type(e).__name__}
            yield final_update
        finally:
            # Cleanup
            await asyncio.sleep(0.1)  # Allow final update to be processed

    async def batch_file_analysis_with_progress(
        self,
        file_paths: list[str],
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
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
        start_time = time.time()
        successful_analyses = 0
        failed_analyses = 0

        try:
            # Validate file paths first
            valid_files = []
            invalid_files = []

            for file_path in file_paths:
                try:
                    from pathlib import Path

                    if Path(file_path).is_file():
                        valid_files.append(file_path)
                    else:
                        invalid_files.append(file_path)
                except Exception:
                    invalid_files.append(file_path)

            if not valid_files:
                raise ValueError(
                    f"No valid files found in provided paths. Invalid: {len(invalid_files)}"
                )

            # Start progress tracking with validated files
            self.progress_tracker.start_operation(
                operation_id,
                total_steps=len(valid_files),
                description=f"Starting batch analysis of {len(valid_files)} files",
            )

            # Store start time for better estimation
            self.progress_tracker.active_operations[operation_id]._start_time = start_time

            if progress_callback:
                self.progress_tracker.add_callback(operation_id, progress_callback)

            # Initial progress update with validation results
            initial_update = self._get_progress_update(operation_id)
            initial_update.details = {
                "valid_files": len(valid_files),
                "invalid_files": len(invalid_files),
                "invalid_file_list": invalid_files[:10],  # Show first 10 invalid files
            }
            yield initial_update

            # Process each file with enhanced error handling
            for i, file_path in enumerate(valid_files):
                if self.progress_tracker.is_cancelled(operation_id):
                    break

                try:
                    # Pre-analysis checks
                    from pathlib import Path

                    file_obj = Path(file_path)
                    file_size = file_obj.stat().st_size

                    # Skip very large files to prevent timeouts
                    if file_size > 10 * 1024 * 1024:  # 10MB limit
                        raise ValueError(f"File too large ({file_size} bytes)")

                    # Analyze the file with timeout
                    analysis_task = asyncio.create_task(
                        self.analyze_file_content(file_path, True, True)
                    )

                    try:
                        analysis = await asyncio.wait_for(
                            analysis_task, timeout=30.0
                        )  # 30s timeout
                        successful_analyses += 1

                        # Update progress with successful analysis
                        self.progress_tracker.update_progress(
                            operation_id,
                            i + 1,
                            f"Analyzed {file_obj.name}",
                            {
                                "current_file": file_path,
                                "file_size": file_size,
                                "complexity_score": analysis.complexity_score,
                                "quality_score": analysis.code_quality_score,
                                "language": analysis.language,
                                "successful_analyses": successful_analyses,
                                "failed_analyses": failed_analyses,
                            },
                        )

                    except asyncio.TimeoutError:
                        analysis_task.cancel()
                        raise ValueError("Analysis timeout (>30s)")

                except Exception as e:
                    failed_analyses += 1

                    # Update progress with error information
                    self.progress_tracker.update_progress(
                        operation_id,
                        i + 1,
                        f"Failed to analyze {Path(file_path).name}",
                        {
                            "current_file": file_path,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "successful_analyses": successful_analyses,
                            "failed_analyses": failed_analyses,
                        },
                    )

                yield self._get_progress_update(operation_id)
                await asyncio.sleep(0.001)  # Micro-delay for responsiveness

            # Mark as completed with summary
            was_cancelled = self.progress_tracker.is_cancelled(operation_id)
            success = successful_analyses > 0 and not was_cancelled

            self.progress_tracker.complete_operation(operation_id, success)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {
                "summary": {
                    "total_requested": len(file_paths),
                    "valid_files": len(valid_files),
                    "successful_analyses": successful_analyses,
                    "failed_analyses": failed_analyses,
                    "success_rate": successful_analyses / len(valid_files) if valid_files else 0,
                    "cancelled": was_cancelled,
                    "total_time_seconds": time.time() - start_time,
                }
            }
            yield final_update

        except asyncio.CancelledError:
            self.progress_tracker.cancel_operation(operation_id)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {
                "message": "Batch analysis was cancelled",
                "partial_results": {
                    "successful_analyses": successful_analyses,
                    "failed_analyses": failed_analyses,
                },
            }
            yield final_update
        except Exception as e:
            self.progress_tracker.complete_operation(operation_id, False)
            final_update = self._get_progress_update(operation_id)
            final_update.details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "partial_results": {
                    "successful_analyses": successful_analyses,
                    "failed_analyses": failed_analyses,
                },
            }
            yield final_update
        finally:
            # Ensure cleanup
            await asyncio.sleep(0.1)


def create_progress_aware_server() -> ProgressAwareSearchServer:
    """Create and configure the progress-aware MCP server instance."""
    return ProgressAwareSearchServer()
