"""History & Utility Tools — search history, analytics, health, and management."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_history_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register history and utility tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    # Import engine-level flags for health check
    from ..engine import SEMANTIC_AVAILABLE

    @mcp.tool
    def get_search_history(limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent search history.

        Args:
            limit: Maximum number of history entries to return (default: 10)

        Returns:
            List of recent search operations with metadata
        """
        try:
            return engine.get_search_history(limit)
        except Exception as e:
            raise ToolError(f"Failed to get history: {e}") from e

    @mcp.tool
    def export_search_history(
        fmt: str = "json",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """
        Export search history as a string.

        Args:
            fmt: Export format - 'json' or 'csv' (default: 'json')
            start_date: Optional start date filter (ISO format YYYY-MM-DD)
            end_date: Optional end date filter (ISO format YYYY-MM-DD)

        Returns:
            Exported history data as string
        """
        try:
            eng = engine._get_engine()
            start_ts = None
            end_ts = None
            if start_date:
                from datetime import datetime as dt

                start_ts = dt.strptime(start_date, "%Y-%m-%d").timestamp()
            if end_date:
                from datetime import datetime as dt

                end_ts = (
                    dt.strptime(end_date, "%Y-%m-%d")
                    .replace(hour=23, minute=59, second=59)
                    .timestamp()
                )
            return eng.export_history_to_string(fmt, start_time=start_ts, end_time=end_ts)
        except Exception as e:
            raise ToolError(f"Failed to export history: {e}") from e

    @mcp.tool
    def search_history_advanced(
        query: str | None = None,
        category: str | None = None,
        language: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Advanced search in history with filtering.

        Args:
            query: Full-text search in query patterns
            category: Filter by category (function/class/variable/import/comment/string/regex/general)
            language: Filter by programming language (python/javascript/typescript/etc.)
            limit: Maximum results to return (default: 20)

        Returns:
            List of matching history entries
        """
        try:
            eng = engine._get_engine()
            if query:
                entries = eng.search_in_history(query, limit)
            elif category:
                entries = eng.get_history_by_category(category, limit)
            elif language:
                entries = eng.get_history_by_language(language, limit)
            else:
                entries = eng.get_search_history(limit)

            from dataclasses import asdict

            result = []
            for entry in entries:
                d = asdict(entry)
                if isinstance(d.get("languages"), set):
                    d["languages"] = sorted(d["languages"])
                if isinstance(d.get("tags"), set):
                    d["tags"] = sorted(d["tags"])
                if hasattr(d.get("category"), "value"):
                    d["category"] = d["category"].value
                result.append(d)
            return result
        except Exception as e:
            raise ToolError(f"Failed to search history: {e}") from e

    @mcp.tool
    def get_history_analytics(days: int = 30) -> dict[str, Any]:
        """
        Get comprehensive search analytics including trends and insights.

        Args:
            days: Number of days to analyze (default: 30)

        Returns:
            Dictionary with analytics, trends, performance insights, and usage patterns
        """
        try:
            eng = engine._get_engine()
            return {
                "analytics": eng.get_search_analytics(days),
                "trends": eng.get_search_trends(days),
                "performance_insights": eng.get_performance_insights(),
                "usage_patterns": eng.get_usage_patterns(),
                "detailed_stats": eng.get_detailed_history_stats(),
            }
        except Exception as e:
            raise ToolError(f"Failed to get analytics: {e}") from e

    @mcp.tool
    def get_failed_patterns(limit: int = 10) -> list[dict[str, Any]]:
        """
        Get search patterns that most frequently return zero results.

        Args:
            limit: Maximum number of patterns to return (default: 10)

        Returns:
            List of failed pattern info with failure counts and rates
        """
        try:
            eng = engine._get_engine()
            return eng.get_top_failed_patterns(limit)
        except Exception as e:
            raise ToolError(f"Failed to get failed patterns: {e}") from e

    @mcp.tool
    def get_session_detail(
        session_id: str | None = None,
        compare_with: str | None = None,
    ) -> dict[str, Any]:
        """
        Get session details or compare two sessions.

        Args:
            session_id: Session ID to get summary for
            compare_with: Optional second session ID to compare with

        Returns:
            Session summary or comparison data
        """
        try:
            eng = engine._get_engine()
            if session_id and compare_with:
                return eng.compare_sessions(session_id, compare_with)
            elif session_id:
                return eng.get_session_summary(session_id)
            else:
                return {"error": "session_id is required"}
        except Exception as e:
            raise ToolError(f"Failed to get session detail: {e}") from e

    @mcp.tool
    def manage_history(
        action: str,
        days: int | None = None,
    ) -> dict[str, Any]:
        """
        Manage search history: cleanup old entries or deduplicate.

        Args:
            action: Action to perform - 'cleanup', 'deduplicate', or 'stats'
            days: For cleanup action, remove entries older than this many days

        Returns:
            Result of the management action
        """
        try:
            eng = engine._get_engine()
            if action == "cleanup":
                if days is None:
                    return {"error": "days parameter is required for cleanup action"}
                removed = eng.cleanup_old_history(days)
                return {"action": "cleanup", "removed_entries": removed, "days": days}
            elif action == "deduplicate":
                removed = eng.deduplicate_history()
                return {"action": "deduplicate", "removed_entries": removed}
            elif action == "stats":
                return eng.get_detailed_history_stats()
            else:
                return {
                    "error": f"Unknown action: {action}. Use 'cleanup', 'deduplicate', or 'stats'"
                }
        except Exception as e:
            raise ToolError(f"Failed to manage history: {e}") from e

    @mcp.tool
    def get_server_health() -> dict[str, Any]:
        """
        Get server health status and diagnostics.

        Returns:
            Comprehensive health status including cache analytics, validation stats,
            session analytics, active operations, and memory usage
        """
        try:
            cache_health = engine.resource_manager.get_health_status()
            validation_stats = engine.validator.get_validation_stats()
            session_analytics = engine.session_manager.get_session_analytics()
            memory_usage = engine.resource_manager.get_memory_usage()
            active_ops = {
                op_id: {
                    "status": op.status.value,
                    "progress": op.progress,
                    "current_step": op.current_step,
                    "elapsed_time": op.elapsed_time,
                }
                for op_id, op in engine.progress_tracker.active_operations.items()
                if op.status.value == "running"
            }
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "cache_health": cache_health,
                "memory_usage": memory_usage,
                "validation_stats": validation_stats,
                "session_analytics": session_analytics,
                "active_operations": active_ops,
                "active_operations_count": len(active_ops),
                "search_history_count": len(engine.search_history),
                "fastmcp_available": True,
                "semantic_available": SEMANTIC_AVAILABLE,
            }
        except Exception as e:
            raise ToolError(f"Health check failed: {e}") from e
