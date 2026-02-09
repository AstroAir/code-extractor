"""Workspace Management Tools — open, create, save, discover, status, search."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_workspace_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register workspace management tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def workspace_open(
        config_path: str,
    ) -> dict[str, Any]:
        """
        Open a workspace from a configuration file (.pysearch-workspace.toml).

        Loads the workspace config, enables multi-repo search, and registers
        all enabled repositories for cross-repo searching.

        Args:
            config_path: Path to .pysearch-workspace.toml file

        Returns:
            Workspace summary with repository count and status
        """
        _validate(paths=[config_path])
        try:
            eng = engine._get_engine()
            success = eng.open_workspace(config_path)
            if not success:
                raise ToolError(f"Failed to open workspace from {config_path}")
            summary = eng.get_workspace_summary()
            return {"opened": True, "summary": summary}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to open workspace: {e}") from e

    @mcp.tool
    def workspace_create(
        name: str,
        root_path: str,
        description: str = "",
        auto_discover: bool = False,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        Create a new workspace and optionally auto-discover Git repositories.

        Args:
            name: Workspace name
            root_path: Root directory for the workspace
            description: Optional description
            auto_discover: Whether to auto-discover Git repos in root_path
            max_depth: Max directory depth for auto-discovery

        Returns:
            Creation status and discovered repositories if auto_discover is True
        """
        _validate(paths=[root_path])
        try:
            eng = engine._get_engine()
            success = eng.create_workspace(
                name,
                root_path,
                description,
                auto_discover=auto_discover,
                max_depth=max_depth,
            )
            if not success:
                raise ToolError("Failed to create workspace")
            result: dict[str, Any] = {
                "created": True,
                "name": name,
                "root_path": root_path,
            }
            if auto_discover:
                result["summary"] = eng.get_workspace_summary()
            return result
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to create workspace: {e}") from e

    @mcp.tool
    def workspace_save(
        config_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Save the current workspace configuration to a TOML file.

        Args:
            config_path: Output path (optional, defaults to workspace root)

        Returns:
            Save status and path
        """
        try:
            eng = engine._get_engine()
            success = eng.save_workspace(config_path)
            return {"saved": success, "path": config_path or "default"}
        except Exception as e:
            raise ToolError(f"Failed to save workspace: {e}") from e

    @mcp.tool
    def workspace_discover(
        root_path: str,
        max_depth: int = 3,
        auto_add: bool = True,
    ) -> dict[str, Any]:
        """
        Discover Git repositories in a directory tree.

        Scans subdirectories for Git repositories, detects project types
        (Python, Node, Java, Go, Rust, etc.), and optionally adds them.

        Args:
            root_path: Directory to scan
            max_depth: Maximum directory depth to search
            auto_add: Whether to automatically add discovered repos

        Returns:
            List of discovered repositories with names, paths, and project types
        """
        _validate(paths=[root_path])
        try:
            eng = engine._get_engine()
            repos = eng.discover_repositories(root_path, max_depth=max_depth, auto_add=auto_add)
            return {
                "discovered": len(repos),
                "repositories": repos,
                "auto_added": auto_add,
            }
        except Exception as e:
            raise ToolError(f"Failed to discover repositories: {e}") from e

    @mcp.tool
    def workspace_status() -> dict[str, Any]:
        """
        Get workspace status summary.

        Returns:
            Workspace summary including repository counts, types, and search settings
        """
        try:
            eng = engine._get_engine()
            summary = eng.get_workspace_summary()
            if not summary:
                return {"loaded": False, "message": "No workspace loaded"}
            return {"loaded": True, "summary": summary}
        except Exception as e:
            raise ToolError(f"Failed to get workspace status: {e}") from e

    @mcp.tool
    def workspace_search(
        pattern: str,
        use_regex: bool = False,
        max_results: int = 1000,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Search across all repositories in the current workspace.

        Args:
            pattern: Search pattern
            use_regex: Whether to use regex matching
            max_results: Maximum number of results
            timeout: Timeout per repository in seconds

        Returns:
            Search results from all workspace repositories
        """
        _validate(pattern=pattern)
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"error": "No workspace/multi-repo enabled. Open a workspace first."}
            result = eng.search_all_repositories(
                pattern=pattern,
                use_regex=use_regex,
                max_results=max_results,
                timeout=timeout,
            )
            if not result:
                return {"total_matches": 0, "repositories_searched": 0}
            return {
                "total_matches": result.total_matches,
                "repositories_searched": result.successful_repositories,
                "total_repositories": result.total_repositories,
                "failed_repositories": result.failed_repositories,
                "search_time_ms": result.search_time_ms,
                "results": {
                    repo_name: {
                        "items": [
                            {
                                "file": str(item.file),
                                "start_line": item.start_line,
                                "end_line": item.end_line,
                                "lines": item.lines,
                            }
                            for item in repo_result.items[:50]
                        ],
                        "total_items": repo_result.stats.items,
                    }
                    for repo_name, repo_result in result.repository_results.items()
                },
            }
        except Exception as e:
            raise ToolError(f"Workspace search failed: {e}") from e
