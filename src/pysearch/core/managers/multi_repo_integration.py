"""
Multi-repository search integration.

This module provides support for searching across multiple repositories
and managing multi-repository search operations.

Classes:
    MultiRepoIntegrationManager: Manages multi-repository search functionality

Key Features:
    - Search across multiple repositories
    - Repository-specific configuration
    - Aggregated search results
    - Repository metadata tracking

Example:
    Using multi-repo search:
        >>> from pysearch.core.managers.multi_repo_integration import MultiRepoIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = MultiRepoIntegrationManager(config)
        >>> manager.enable_multi_repo()
        >>> # Multi-repository search is now available
"""

from __future__ import annotations

from typing import Any

from ..config import SearchConfig


class MultiRepoIntegrationManager:
    """Manages multi-repository search functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.multi_repo_engine: Any = None
        self._multi_repo_enabled = False

    def enable_multi_repo(self, repositories: list[dict[str, Any]] | None = None) -> bool:
        """
        Enable multi-repository search functionality.

        Args:
            repositories: List of repository configurations

        Returns:
            True if multi-repo was enabled successfully, False otherwise
        """
        if self._multi_repo_enabled:
            return True

        try:
            from ...integrations.multi_repo import MultiRepoSearchEngine

            self.multi_repo_engine = MultiRepoSearchEngine()

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
            return self.multi_repo_engine.add_repository(name, path, **kwargs)  # type: ignore[no-any-return]
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
            return self.multi_repo_engine.remove_repository(name)  # type: ignore[no-any-return]
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
            return self.multi_repo_engine.list_repositories()  # type: ignore[no-any-return]
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
            return self.multi_repo_engine.get_search_statistics()  # type: ignore[no-any-return]
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
            return self.multi_repo_engine.get_health_status()  # type: ignore[no-any-return]
        except Exception:
            return {}
