"""Multi-Repository Search Tools — enable, add, remove, list, search, health, stats, sync."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_multi_repo_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register multi-repository search tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def multi_repo_enable(
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """
        Enable multi-repository search capabilities.

        Args:
            max_workers: Maximum number of parallel workers for searches

        Returns:
            Status of multi-repo enablement
        """
        try:
            eng = engine._get_engine()
            success = eng.enable_multi_repo(max_workers=max_workers)
            return {"enabled": success, "max_workers": max_workers}
        except Exception as e:
            raise ToolError(f"Failed to enable multi-repo: {e}") from e

    @mcp.tool
    def multi_repo_add(
        name: str,
        path: str,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """
        Add a repository to multi-repository search.

        Args:
            name: Unique name for the repository
            path: Filesystem path to the repository
            priority: Priority level — "high", "normal", "low"

        Returns:
            Result of adding the repository
        """
        _validate(paths=[path])
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                eng.enable_multi_repo()
            success = eng.add_repository(name, path, priority=priority)
            return {"added": success, "name": name, "path": path, "priority": priority}
        except Exception as e:
            raise ToolError(f"Failed to add repository: {e}") from e

    @mcp.tool
    def multi_repo_remove(
        name: str,
    ) -> dict[str, Any]:
        """
        Remove a repository from multi-repository search.

        Args:
            name: Name of the repository to remove

        Returns:
            Result of removing the repository
        """
        try:
            eng = engine._get_engine()
            success = eng.remove_repository(name)
            return {"removed": success, "name": name}
        except Exception as e:
            raise ToolError(f"Failed to remove repository: {e}") from e

    @mcp.tool
    def multi_repo_list() -> dict[str, Any]:
        """
        List all repositories in the multi-repository search system.

        Returns:
            List of repository names and enabled status
        """
        try:
            eng = engine._get_engine()
            enabled = eng.is_multi_repo_enabled()
            repos = eng.list_repositories() if enabled else []
            return {"enabled": enabled, "repositories": repos, "count": len(repos)}
        except Exception as e:
            raise ToolError(f"Failed to list repositories: {e}") from e

    @mcp.tool
    def multi_repo_search(
        pattern: str,
        use_regex: bool = False,
        use_ast: bool = False,
        context: int = 2,
        max_results: int = 1000,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Search across all repositories in the multi-repository system.

        Args:
            pattern: Search pattern
            use_regex: Whether to use regex matching
            use_ast: Whether to use AST-based matching
            context: Number of context lines around matches
            max_results: Maximum total results
            timeout: Timeout per repository search in seconds

        Returns:
            Aggregated search results from all repositories
        """
        _validate(pattern=pattern)
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"error": "Multi-repo not enabled. Call multi_repo_enable first."}
            result = eng.search_all_repositories(
                pattern=pattern,
                use_regex=use_regex,
                use_ast=use_ast,
                context=context,
                max_results=max_results,
                timeout=timeout,
            )
            if result is None:
                return {"total_matches": 0, "repositories": {}}
            return {
                "total_matches": result.total_matches,
                "total_repositories": result.total_repositories,
                "successful_repositories": result.successful_repositories,
                "failed_repositories": result.failed_repositories,
                "success_rate": result.success_rate,
                "repository_results": {
                    name: {
                        "items": len(sr.items),
                        "files_matched": sr.stats.files_matched,
                        "elapsed_ms": sr.stats.elapsed_ms,
                    }
                    for name, sr in result.repository_results.items()
                },
            }
        except Exception as e:
            raise ToolError(f"Multi-repo search failed: {e}") from e

    @mcp.tool
    def multi_repo_health() -> dict[str, Any]:
        """
        Get health status for all repositories in the multi-repository system.

        Returns:
            Health information including per-repository status and overall summary
        """
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"enabled": False, "message": "Multi-repo not enabled"}
            return eng.get_multi_repo_health()
        except Exception as e:
            raise ToolError(f"Failed to get multi-repo health: {e}") from e

    @mcp.tool
    def multi_repo_stats() -> dict[str, Any]:
        """
        Get search performance statistics for the multi-repository system.

        Returns:
            Statistics including total searches, average search time, and pattern history
        """
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"enabled": False, "message": "Multi-repo not enabled"}
            return eng.get_multi_repo_stats()
        except Exception as e:
            raise ToolError(f"Failed to get multi-repo stats: {e}") from e

    @mcp.tool
    def multi_repo_sync() -> dict[str, Any]:
        """
        Synchronize all repositories (refresh status and health).

        Returns:
            Dictionary mapping repository names to sync success status
        """
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"enabled": False, "message": "Multi-repo not enabled"}
            return eng.sync_repositories()
        except Exception as e:
            raise ToolError(f"Failed to sync repositories: {e}") from e
