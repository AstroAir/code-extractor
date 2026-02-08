"""Progress Tracking Tools — query and cancel running operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_progress_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register progress tracking tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def get_operation_progress(
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get progress of active operations.

        If operation_id is provided, returns progress for that specific operation.
        Otherwise, returns a summary of all active operations.

        Args:
            operation_id: Optional specific operation ID to query

        Returns:
            Progress information including status, completion percentage, and timing
        """
        try:
            if operation_id:
                _validate(operation_id=operation_id)
                status = engine.progress_tracker.get_operation_status(operation_id)
                if status is None:
                    return {"error": f"Operation not found: {operation_id}"}
                return {
                    "operation_id": status.operation_id,
                    "status": status.status.value,
                    "progress": status.progress,
                    "current_step": status.current_step,
                    "total_steps": status.total_steps,
                    "completed_steps": status.completed_steps,
                    "elapsed_time": status.elapsed_time,
                    "estimated_remaining": status.estimated_remaining,
                    "details": status.details,
                }
            else:
                ops = {}
                for op_id, op in engine.progress_tracker.active_operations.items():
                    ops[op_id] = {
                        "status": op.status.value,
                        "progress": op.progress,
                        "current_step": op.current_step,
                        "elapsed_time": op.elapsed_time,
                    }
                return {
                    "active_operations_count": len(ops),
                    "operations": ops,
                }
        except Exception as e:
            raise ToolError(f"Failed to get operation progress: {e}") from e

    @mcp.tool
    def cancel_operation(
        operation_id: str,
    ) -> dict[str, Any]:
        """
        Cancel a running operation.

        Args:
            operation_id: The operation ID to cancel

        Returns:
            Cancellation status
        """
        _validate(operation_id=operation_id)
        try:
            success = engine.progress_tracker.cancel_operation(operation_id)
            if success:
                return {
                    "operation_id": operation_id,
                    "status": "cancelled",
                    "message": f"Operation {operation_id} has been cancelled",
                }
            else:
                return {
                    "operation_id": operation_id,
                    "status": "not_found",
                    "message": f"Operation {operation_id} not found or already completed",
                }
        except Exception as e:
            raise ToolError(f"Failed to cancel operation: {e}") from e
