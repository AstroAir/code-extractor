"""Configuration Tools — search configuration management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_config_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register configuration tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def configure_search(
        paths: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        context: int | None = None,
        parallel: bool | None = None,
        workers: int | None = None,
        languages: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Update search configuration.

        Args:
            paths: List of paths to search
            include_patterns: File patterns to include (e.g., ["**/*.py", "**/*.js"])
            exclude_patterns: File patterns to exclude (e.g., ["**/node_modules/**"])
            context: Number of context lines around matches
            parallel: Whether to use parallel processing
            workers: Number of worker threads
            languages: List of languages to filter by

        Returns:
            Updated configuration settings
        """
        try:
            resp = engine.configure_search(
                paths,
                include_patterns,
                exclude_patterns,
                context,
                parallel,
                workers,
                languages,
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Configuration update failed: {e}") from e

    @mcp.tool
    def get_search_config() -> dict[str, Any]:
        """
        Get current search configuration.

        Returns:
            Current configuration settings including paths, patterns, and options
        """
        try:
            resp = engine.get_search_config()
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Failed to get configuration: {e}") from e

    @mcp.tool
    def get_supported_languages() -> list[str]:
        """
        Get list of supported programming languages.

        Returns:
            List of supported language names
        """
        try:
            return engine.get_supported_languages()
        except Exception as e:
            raise ToolError(f"Failed to get languages: {e}") from e

    @mcp.tool
    def clear_caches() -> dict[str, Any]:
        """
        Clear search engine caches, optimize resources, and cleanup stale sessions.

        Returns:
            Detailed status including cache analytics before clearing, sessions cleaned,
            and progress operations cleaned
        """
        try:
            return engine.clear_caches()
        except Exception as e:
            raise ToolError(f"Failed to clear caches: {e}") from e
