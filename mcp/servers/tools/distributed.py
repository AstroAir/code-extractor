"""Distributed Indexing Tools — enable, disable, and query distributed indexing status."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_distributed_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register distributed indexing tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def distributed_enable(
        num_workers: int | None = None,
        max_queue_size: int = 10000,
    ) -> dict[str, Any]:
        """
        Enable distributed indexing for large codebases.

        Args:
            num_workers: Number of worker processes (defaults to min(cpu_count, 8))
            max_queue_size: Maximum work queue size

        Returns:
            Status of distributed indexing enablement
        """
        try:
            eng = engine._get_engine()
            success = eng.enable_distributed_indexing(
                num_workers=num_workers, max_queue_size=max_queue_size
            )
            return {
                "enabled": success,
                "num_workers": num_workers or "auto",
                "max_queue_size": max_queue_size,
            }
        except Exception as e:
            raise ToolError(f"Failed to enable distributed indexing: {e}") from e

    @mcp.tool
    def distributed_disable() -> dict[str, Any]:
        """
        Disable distributed indexing and stop all workers.

        Returns:
            Confirmation of distributed indexing disablement
        """
        try:
            eng = engine._get_engine()
            eng.disable_distributed_indexing()
            return {"enabled": False, "message": "Distributed indexing disabled"}
        except Exception as e:
            raise ToolError(f"Failed to disable distributed indexing: {e}") from e

    @mcp.tool
    def distributed_status() -> dict[str, Any]:
        """
        Get the current status of distributed indexing including worker and queue stats.

        Returns:
            Comprehensive status including enabled state, queue stats, and worker info
        """
        try:
            eng = engine._get_engine()
            enabled = eng.is_distributed_indexing_enabled()
            result: dict[str, Any] = {"enabled": enabled}
            if enabled:
                result["queue_stats"] = eng.get_distributed_queue_stats()
            return result
        except Exception as e:
            raise ToolError(f"Failed to get distributed status: {e}") from e
