"""MCP Resource Endpoints — configuration, history, stats, sessions, languages."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from .engine import PySearchEngine


def register_resources(
    mcp: FastMCP,
    engine: PySearchEngine,
) -> None:
    """Register all MCP resource endpoints on the server."""

    @mcp.resource("pysearch://config/current")
    async def resource_current_config() -> str:
        """Get current search configuration as JSON."""
        try:
            resp = engine.get_search_config()
            return json.dumps(asdict(resp), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://history/searches")
    async def resource_search_history() -> str:
        """Get complete search history as JSON."""
        try:
            return json.dumps(engine.get_search_history(100), indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://stats/overview")
    async def resource_stats_overview() -> str:
        """Get comprehensive statistics overview as JSON."""
        try:
            cache_analytics = engine.resource_manager.get_cache_analytics()
            memory_usage = engine.resource_manager.get_memory_usage()
            session_analytics = engine.session_manager.get_session_analytics()
            active_ops_count = len(
                [
                    op
                    for op in engine.progress_tracker.active_operations.values()
                    if op.status.value == "running"
                ]
            )
            stats = {
                "total_searches": len(engine.search_history),
                "cache_analytics": cache_analytics,
                "memory_usage": memory_usage,
                "validation_stats": engine.validator.get_validation_stats(),
                "session_analytics": session_analytics,
                "active_operations_count": active_ops_count,
                "timestamp": datetime.now().isoformat(),
            }
            return json.dumps(stats, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://sessions/analytics")
    async def resource_sessions_analytics() -> str:
        """Get session management analytics as JSON."""
        try:
            analytics = engine.session_manager.get_session_analytics()
            return json.dumps(analytics, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://languages/supported")
    async def resource_supported_languages() -> str:
        """Get supported languages as JSON."""
        try:
            langs = engine.get_supported_languages()
            return json.dumps({"languages": langs, "count": len(langs)}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
