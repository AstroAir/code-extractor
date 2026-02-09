"""
Multi-repository search integration.

This module provides support for searching across multiple repositories
and managing multi-repository search operations, including workspace-level
configuration and repository auto-discovery.

Classes:
    MultiRepoIntegrationManager: Manages multi-repository search functionality

Key Features:
    - Search across multiple repositories
    - Repository-specific configuration
    - Aggregated search results
    - Repository metadata tracking
    - Workspace configuration persistence
    - Repository auto-discovery

Example:
    Using multi-repo search:
        >>> from pysearch.core.managers.multi_repo_integration import MultiRepoIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = MultiRepoIntegrationManager(config)
        >>> manager.enable_multi_repo()
        >>> # Multi-repository search is now available

    Using workspace support:
        >>> manager.load_workspace("/path/to/.pysearch-workspace.toml")
        >>> repos = manager.list_repositories()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import SearchConfig


class MultiRepoIntegrationManager:
    """Manages multi-repository search functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.multi_repo_engine: Any = None
        self._multi_repo_enabled = False
        self._workspace_config: Any = None
        self._workspace_manager: Any = None

    def enable_multi_repo(
        self,
        repositories: list[dict[str, Any]] | None = None,
        max_workers: int = 4,
    ) -> bool:
        """
        Enable multi-repository search functionality.

        Args:
            repositories: List of repository configurations
            max_workers: Maximum number of parallel workers for searches

        Returns:
            True if multi-repo was enabled successfully, False otherwise
        """
        if self._multi_repo_enabled:
            return True

        try:
            from ...integrations.multi_repo import MultiRepoSearchEngine

            self.multi_repo_engine = MultiRepoSearchEngine(max_workers=max_workers)

            if repositories and self.multi_repo_engine:
                for repo_config in repositories:
                    self.multi_repo_engine.add_repository(**repo_config)

            self._multi_repo_enabled = True
            return True

        except Exception:
            return False

    def disable_multi_repo(self) -> None:
        """Disable multi-repository search functionality."""
        if not self._multi_repo_enabled:
            return

        self.multi_repo_engine = None
        self._multi_repo_enabled = False

    def is_multi_repo_enabled(self) -> bool:
        """Check if multi-repository search is enabled."""
        return self._multi_repo_enabled

    def add_repository(self, name: str, path: str, **kwargs: Any) -> bool:
        """
        Add a repository to the multi-repo search.

        Args:
            name: Repository name
            path: Repository path
            **kwargs: Additional repository configuration

        Returns:
            True if repository was added successfully, False otherwise
        """
        if not self.multi_repo_engine:
            return False

        try:
            return self.multi_repo_engine.add_repository(name, path, **kwargs)
        except Exception:
            return False

    def configure_repository(self, name: str, **config_updates: Any) -> bool:
        """
        Update repository configuration.

        Args:
            name: Repository name
            **config_updates: Configuration updates (priority, enabled, etc.)

        Returns:
            True if configuration was updated, False otherwise
        """
        if not self.multi_repo_engine:
            return False

        try:
            return self.multi_repo_engine.configure_repository(name, **config_updates)
        except Exception:
            return False

    def remove_repository(self, name: str) -> bool:
        """
        Remove a repository from multi-repo search.

        Args:
            name: Repository name to remove

        Returns:
            True if repository was removed successfully, False otherwise
        """
        if not self.multi_repo_engine:
            return False

        try:
            return self.multi_repo_engine.remove_repository(name)
        except Exception:
            return False

    def list_repositories(self) -> list[str]:
        """
        Get list of configured repository names.

        Returns:
            List of repository names
        """
        if not self.multi_repo_engine:
            return []

        try:
            return self.multi_repo_engine.list_repositories()
        except Exception:
            return []

    def search_repositories(self, query: Any, repositories: list[str] | None = None) -> Any:
        """
        Search across multiple repositories.

        Args:
            query: Search query
            repositories: Specific repositories to search (None for all)

        Returns:
            Multi-repository search results
        """
        if not self.multi_repo_engine:
            return None

        try:
            return self.multi_repo_engine.search_repositories(
                repositories=repositories,
                query=query,
            )
        except Exception:
            return None

    def get_repository_stats(self) -> dict[str, Any]:
        """
        Get statistics for all repositories.

        Returns:
            Dictionary with repository statistics
        """
        if not self.multi_repo_engine:
            return {}

        try:
            return self.multi_repo_engine.get_search_statistics()
        except Exception:
            return {}

    def sync_repositories(self) -> dict[str, bool]:
        """
        Synchronize all repositories (refresh status and health).

        Returns:
            Dictionary mapping repository names to sync success status
        """
        if not self.multi_repo_engine:
            return {}

        try:
            self.multi_repo_engine.repository_manager.refresh_all_status()
            results: dict[str, bool] = {}
            for name in self.multi_repo_engine.list_repositories():
                repo = self.multi_repo_engine.get_repository_info(name)
                results[name] = repo.health_status == "healthy" if repo else False
            return results
        except Exception:
            return {}

    def get_repository_health(self) -> dict[str, dict[str, Any]]:
        """
        Get health status for all repositories.

        Returns:
            Dictionary mapping repository names to health information
        """
        if not self.multi_repo_engine:
            return {}

        try:
            return self.multi_repo_engine.get_health_status()
        except Exception:
            return {}

    # -- Workspace Support --------------------------------------------------

    def _ensure_workspace_manager(self) -> Any:
        """Lazy-load the WorkspaceManager."""
        if self._workspace_manager is None:
            from ..workspace import WorkspaceManager

            self._workspace_manager = WorkspaceManager()
        return self._workspace_manager

    def load_workspace(self, config_path: str | Path) -> bool:
        """
        Load a workspace configuration and initialize all repositories.

        This method:
        1. Parses the workspace TOML config
        2. Enables multi-repo search if not already enabled
        3. Adds all enabled repositories from the workspace

        Args:
            config_path: Path to .pysearch-workspace.toml

        Returns:
            True if workspace was loaded successfully, False otherwise
        """
        try:
            mgr = self._ensure_workspace_manager()
            ws = mgr.load_workspace(config_path)
            self._workspace_config = ws

            # Enable multi-repo if needed
            if not self._multi_repo_enabled:
                if not self.enable_multi_repo(max_workers=ws.max_workers):
                    return False

            # Add all enabled repositories
            for repo_cfg in ws.get_enabled_repositories():
                repo_path = Path(repo_cfg.path)
                if not repo_path.is_absolute():
                    repo_path = Path(ws.root_path) / repo_path

                # Build per-repo SearchConfig with workspace defaults + overrides
                include = repo_cfg.include or ws.include
                exclude = repo_cfg.exclude or ws.exclude

                search_cfg = SearchConfig(
                    paths=[str(repo_path)],
                    include=include,
                    exclude=exclude,
                    context=ws.context,
                    parallel=ws.parallel,
                    workers=ws.workers,
                    follow_symlinks=ws.follow_symlinks,
                )

                self.add_repository(
                    name=repo_cfg.name,
                    path=str(repo_path),
                    config=search_cfg,
                    priority=repo_cfg.priority,
                )

            return True

        except Exception:
            return False

    def save_workspace(self, config_path: str | Path | None = None) -> bool:
        """
        Save the current workspace configuration to a TOML file.

        Args:
            config_path: Output path. If None, uses the workspace root path.

        Returns:
            True if saved successfully, False otherwise
        """
        if not self._workspace_config:
            return False

        try:
            mgr = self._ensure_workspace_manager()
            mgr.save_workspace(self._workspace_config, config_path)
            return True
        except Exception:
            return False

    def get_workspace_config(self) -> Any:
        """
        Get the current workspace configuration.

        Returns:
            WorkspaceConfig if a workspace is loaded, None otherwise
        """
        return self._workspace_config

    def create_workspace(
        self,
        name: str,
        root_path: str | Path,
        description: str = "",
        **metadata: Any,
    ) -> Any:
        """
        Create a new workspace configuration.

        Args:
            name: Workspace name
            root_path: Root directory
            description: Optional description
            **metadata: Arbitrary metadata

        Returns:
            WorkspaceConfig instance
        """
        try:
            mgr = self._ensure_workspace_manager()
            ws = mgr.create_workspace(name, root_path, description, **metadata)
            self._workspace_config = ws
            return ws
        except Exception:
            return None

    def discover_repositories(
        self,
        root_path: str | Path | None = None,
        max_depth: int = 3,
        auto_add: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Discover Git repositories in a directory tree.

        Args:
            root_path: Directory to scan (defaults to workspace root)
            max_depth: Maximum search depth
            auto_add: Whether to auto-add discovered repos to workspace and engine

        Returns:
            List of discovered repository info dicts
        """
        try:
            mgr = self._ensure_workspace_manager()

            # Create workspace if none exists
            if self._workspace_config is None:
                scan_root = str(Path(root_path).resolve()) if root_path else "."
                self._workspace_config = mgr.create_workspace("auto-discovered", scan_root)

            discovered = mgr.discover_repositories(
                self._workspace_config,
                root_path=root_path,
                max_depth=max_depth,
                auto_add=auto_add,
            )

            # If auto_add and multi-repo is enabled, also add to the engine
            if auto_add and self._multi_repo_enabled and self.multi_repo_engine:
                ws = self._workspace_config
                for repo_cfg in discovered:
                    repo_path = Path(repo_cfg.path)
                    include = repo_cfg.include or ws.include
                    exclude = repo_cfg.exclude or ws.exclude

                    search_cfg = SearchConfig(
                        paths=[str(repo_path)],
                        include=include,
                        exclude=exclude,
                        context=ws.context,
                        parallel=ws.parallel,
                        workers=ws.workers,
                    )
                    self.add_repository(
                        name=repo_cfg.name,
                        path=str(repo_path),
                        config=search_cfg,
                        priority=repo_cfg.priority,
                    )

            return [r.to_dict() for r in discovered]

        except Exception:
            return []

    def get_workspace_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current workspace.

        Returns:
            Dictionary with workspace summary, empty if no workspace loaded
        """
        if not self._workspace_config:
            return {}

        try:
            mgr = self._ensure_workspace_manager()
            return mgr.get_workspace_summary(self._workspace_config)
        except Exception:
            return {}
