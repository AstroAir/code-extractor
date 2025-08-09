#!/usr/bin/env python3
"""
MCP Resource Management for PySearch

This module implements MCP resource management functionality for exposing
search indexes, cached results, configuration files, and search history
as accessible resources through the Model Context Protocol.

Resources provided:
- Search configuration
- Search history
- File analysis cache
- Search indexes
- Session data
- Statistics and metrics
"""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

# Import MCP server
from ..servers.mcp_server import (
    PySearchMCPServer,
)


class MCPResourceManager:
    """
    Manages MCP resources for the PySearch server.

    Provides access to various server resources through the MCP protocol:
    - Configuration files
    - Search history and sessions
    - Cached analysis results
    - Search indexes and statistics
    """

    def __init__(self, server: PySearchMCPServer):
        self.server = server
        self.resource_cache: dict[str, Any] = {}
        self.cache_timestamps: dict[str, float] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL

    def get_available_resources(self) -> list[dict[str, Any]]:
        """
        Get list of available MCP resources.

        Returns:
            List of resource descriptors
        """
        return [
            {
                "uri": "pysearch://config/current",
                "name": "Current Search Configuration",
                "description": "Current search configuration including paths, patterns, and settings",
                "mimeType": "application/json",
            },
            {
                "uri": "pysearch://history/searches",
                "name": "Search History",
                "description": "Complete search history with queries, results, and metadata",
                "mimeType": "application/json",
            },
            {
                "uri": "pysearch://sessions/active",
                "name": "Active Search Sessions",
                "description": "Currently active search sessions with context and state",
                "mimeType": "application/json",
            },
            {
                "uri": "pysearch://cache/file-analysis",
                "name": "File Analysis Cache",
                "description": "Cached file analysis results including complexity and quality metrics",
                "mimeType": "application/json",
            },
            {
                "uri": "pysearch://stats/overview",
                "name": "Search Statistics Overview",
                "description": "Comprehensive statistics about searches, files, and performance",
                "mimeType": "application/json",
            },
            {
                "uri": "pysearch://index/languages",
                "name": "Language Index",
                "description": "Index of supported languages and file type mappings",
                "mimeType": "application/json",
            },
            {
                "uri": "pysearch://config/ranking-weights",
                "name": "Ranking Weights Configuration",
                "description": "Current ranking factor weights for search result scoring",
                "mimeType": "application/json",
            },
            {
                "uri": "pysearch://performance/metrics",
                "name": "Performance Metrics",
                "description": "Performance metrics and timing data for search operations",
                "mimeType": "application/json",
            },
        ]

    async def get_resource_content(self, uri: str) -> dict[str, Any]:
        """
        Get content for a specific resource URI.

        Args:
            uri: Resource URI to fetch

        Returns:
            Resource content as dictionary
        """
        # Check cache first
        if self._is_cached(uri):
            cached_content = self.resource_cache[uri]
            return cached_content if isinstance(cached_content, dict) else {}

        content = await self._fetch_resource_content(uri)

        # Cache the result
        self.resource_cache[uri] = content
        self.cache_timestamps[uri] = time.time()

        return content

    def _is_cached(self, uri: str) -> bool:
        """Check if resource is cached and still valid."""
        if uri not in self.resource_cache:
            return False

        cache_time = self.cache_timestamps.get(uri, 0)
        return (time.time() - cache_time) < self.cache_ttl

    async def _fetch_resource_content(self, uri: str) -> dict[str, Any]:
        """Fetch fresh content for a resource URI."""

        if uri == "pysearch://config/current":
            return await self._get_current_config()

        elif uri == "pysearch://history/searches":
            return await self._get_search_history()

        elif uri == "pysearch://sessions/active":
            return await self._get_active_sessions()

        elif uri == "pysearch://cache/file-analysis":
            return await self._get_file_analysis_cache()

        elif uri == "pysearch://stats/overview":
            return await self._get_stats_overview()

        elif uri == "pysearch://index/languages":
            return await self._get_language_index()

        elif uri == "pysearch://config/ranking-weights":
            return await self._get_ranking_weights()

        elif uri == "pysearch://performance/metrics":
            return await self._get_performance_metrics()

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def _get_current_config(self) -> dict[str, Any]:
        """Get current search configuration."""
        if not self.server.current_config:
            return {"error": "No configuration available"}

        config_response = await self.server.get_search_config()

        return {
            "resource_type": "configuration",
            "timestamp": datetime.now().isoformat(),
            "configuration": asdict(config_response),
            "metadata": {
                "server_name": self.server.name,
                "initialized": self.server.search_engine is not None,
            },
        }

    async def _get_search_history(self) -> dict[str, Any]:
        """Get complete search history."""
        return {
            "resource_type": "search_history",
            "timestamp": datetime.now().isoformat(),
            "history": self.server.search_history,
            "metadata": {
                "total_searches": len(self.server.search_history),
                "history_size_limit": 100,
            },
        }

    async def _get_active_sessions(self) -> dict[str, Any]:
        """Get active search sessions."""
        sessions_data = {}

        for session_id, session in self.server.search_sessions.items():
            sessions_data[session_id] = {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "query_count": len(session.queries),
                "context_keys": list(session.context.keys()),
                "cached_results_count": len(session.cached_results),
            }

        return {
            "resource_type": "active_sessions",
            "timestamp": datetime.now().isoformat(),
            "sessions": sessions_data,
            "metadata": {
                "total_sessions": len(self.server.search_sessions),
                "active_sessions": len(
                    [
                        s
                        for s in self.server.search_sessions.values()
                        if (datetime.now() - s.last_accessed).seconds < 3600
                    ]
                ),
            },
        }

    async def _get_file_analysis_cache(self) -> dict[str, Any]:
        """Get file analysis cache contents."""
        cache_data = {}

        for cache_key, analysis in self.server.file_analysis_cache.items():
            cache_data[cache_key] = asdict(analysis)

        return {
            "resource_type": "file_analysis_cache",
            "timestamp": datetime.now().isoformat(),
            "cache": cache_data,
            "metadata": {
                "cached_files": len(self.server.file_analysis_cache),
                "cache_size_mb": self._estimate_cache_size_mb(),
            },
        }

    async def _get_stats_overview(self) -> dict[str, Any]:
        """Get comprehensive statistics overview."""
        try:
            file_stats = await self.server.get_file_statistics(include_analysis=True)
        except Exception as e:
            file_stats = {"error": str(e)}

        return {
            "resource_type": "statistics_overview",
            "timestamp": datetime.now().isoformat(),
            "file_statistics": file_stats,
            "search_statistics": {
                "total_searches": len(self.server.search_history),
                "active_sessions": len(self.server.search_sessions),
                "cached_analyses": len(self.server.file_analysis_cache),
            },
            "performance_summary": self._get_performance_summary(),
        }

    async def _get_language_index(self) -> dict[str, Any]:
        """Get language index and mappings."""
        try:
            supported_languages = await self.server.get_supported_languages()
        except Exception:
            supported_languages = []

        return {
            "resource_type": "language_index",
            "timestamp": datetime.now().isoformat(),
            "supported_languages": supported_languages,
            "language_mappings": self._get_language_mappings(),
            "metadata": {"total_languages": len(supported_languages)},
        }

    async def _get_ranking_weights(self) -> dict[str, Any]:
        """Get current ranking weights configuration."""
        return {
            "resource_type": "ranking_weights",
            "timestamp": datetime.now().isoformat(),
            "weights": {
                factor.value: weight for factor, weight in self.server.ranking_weights.items()
            },
            "metadata": {
                "total_weight": sum(self.server.ranking_weights.values()),
                "factors_count": len(self.server.ranking_weights),
            },
        }

    async def _get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics and timing data."""
        return {
            "resource_type": "performance_metrics",
            "timestamp": datetime.now().isoformat(),
            "metrics": self._calculate_performance_metrics(),
            "cache_performance": {
                "resource_cache_size": len(self.resource_cache),
                "file_analysis_cache_size": len(self.server.file_analysis_cache),
                "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            },
        }

    def _estimate_cache_size_mb(self) -> float:
        """Estimate cache size in MB (simplified)."""
        # Rough estimation based on number of cached items
        return len(self.server.file_analysis_cache) * 0.001  # ~1KB per analysis

    def _get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary from search history."""
        if not self.server.search_history:
            return {"no_data": True}

        execution_times = [
            search.get("execution_time_ms", 0) for search in self.server.search_history
        ]
        result_counts = [search.get("result_count", 0) for search in self.server.search_history]

        return {
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "max_execution_time_ms": max(execution_times),
            "min_execution_time_ms": min(execution_times),
            "avg_results_per_search": sum(result_counts) / len(result_counts),
            "total_searches": len(self.server.search_history),
        }

    def _get_language_mappings(self) -> dict[str, list[str]]:
        """Get file extension to language mappings."""
        return {
            "python": [".py", ".pyw", ".pyi"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "cpp": [".cpp", ".cxx", ".cc", ".c++"],
            "c": [".c", ".h"],
            "csharp": [".cs"],
            "go": [".go"],
            "rust": [".rs"],
            "php": [".php", ".phtml"],
            "ruby": [".rb", ".rbw"],
            "swift": [".swift"],
            "kotlin": [".kt", ".kts"],
            "scala": [".scala", ".sc"],
            "html": [".html", ".htm"],
            "css": [".css", ".scss", ".sass", ".less"],
            "sql": [".sql"],
            "shell": [".sh", ".bash", ".zsh", ".fish"],
        }

    def _calculate_performance_metrics(self) -> dict[str, Any]:
        """Calculate detailed performance metrics."""
        # This would include more sophisticated metrics in a real implementation
        return {
            "uptime_seconds": time.time() - getattr(self.server, "_start_time", time.time()),
            "memory_usage_estimate": "N/A",  # Would need psutil for real memory usage
            "cache_efficiency": self._calculate_cache_hit_ratio(),
        }

    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        # This is a simplified implementation
        # In practice, you'd track cache hits vs misses
        return 0.85  # Placeholder value

    def clear_resource_cache(self) -> dict[str, str]:
        """Clear the resource cache."""
        cache_size = len(self.resource_cache)
        self.resource_cache.clear()
        self.cache_timestamps.clear()

        return {"status": "success", "message": f"Cleared {cache_size} cached resources"}
