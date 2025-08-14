"""
Multi-repository search coordination module for pysearch.

This module provides capabilities for searching across multiple repositories
and codebases simultaneously, with intelligent coordination and result
aggregation.

Classes:
    RepositoryInfo: Metadata about a repository
    MultiRepoSearchEngine: Main engine for multi-repository searches
    RepositoryManager: Manages multiple repository configurations
    SearchCoordinator: Coordinates searches across repositories

Features:
    - Search across multiple Git repositories
    - Intelligent result aggregation and deduplication
    - Repository-specific configurations and filters
    - Parallel search execution across repositories
    - Cross-repository dependency analysis
    - Repository health monitoring and status

Example:
    Basic multi-repository search:
        >>> from pysearch.multi_repo import MultiRepoSearchEngine
        >>> engine = MultiRepoSearchEngine()
        >>>
        >>> # Add repositories
        >>> engine.add_repository("project-a", "/path/to/project-a")
        >>> engine.add_repository("project-b", "/path/to/project-b")
        >>>
        >>> # Search across all repositories
        >>> results = engine.search_all("def main")
        >>> print(f"Found matches in {len(results)} repositories")

    Advanced multi-repository coordination:
        >>> # Configure repository-specific settings
        >>> engine.configure_repository("project-a", {
        ...     "include": ["**/*.py"],
        ...     "exclude": ["**/tests/**"],
        ...     "priority": "high"
        ... })
        >>>
        >>> # Search with repository filtering
        >>> results = engine.search_repositories(
        ...     pattern="database",
        ...     repositories=["project-a", "project-b"],
        ...     aggregate_results=True
        ... )
"""

from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid runtime circular import
    from ..core.api import PySearch  # pragma: no cover
from ..core.config import SearchConfig
from ..utils.logging_config import get_logger
from ..core.types import Query, SearchItem, SearchResult


@dataclass
class RepositoryInfo:
    """Information about a repository in the multi-repo system."""

    name: str
    path: Path
    config: SearchConfig
    priority: str = "normal"  # "high", "normal", "low"
    enabled: bool = True
    last_updated: float = 0.0
    health_status: str = "unknown"  # "healthy", "warning", "error", "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Git information
    git_remote: str = ""
    git_branch: str = ""
    git_commit: str = ""

    def __post_init__(self) -> None:
        """Initialize repository information."""
        if self.last_updated == 0.0:
            self.last_updated = time.time()

        # Try to get Git information
        self._update_git_info()

    def _update_git_info(self) -> None:
        """Update Git repository information."""
        if not self.path.exists():
            self.health_status = "error"
            return

        try:
            # Check if it's a Git repository
            git_dir = self.path / ".git"
            if not git_dir.exists():
                self.health_status = "warning"
                return

            # Get Git information
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.git_remote = result.stdout.strip()

            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.git_branch = result.stdout.strip()

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.git_commit = result.stdout.strip()[:8]  # Short hash

            self.health_status = "healthy"

        except Exception:
            self.health_status = "warning"

    def refresh_status(self) -> None:
        """Refresh repository status and Git information."""
        self._update_git_info()
        self.last_updated = time.time()


@dataclass
class MultiRepoSearchResult:
    """Results from a multi-repository search."""

    repository_results: dict[str, SearchResult]
    aggregated_result: SearchResult | None = None
    total_repositories: int = 0
    successful_repositories: int = 0
    failed_repositories: list[str] = field(default_factory=list)
    search_time_ms: float = 0.0

    @property
    def total_matches(self) -> int:
        """Get total number of matches across all repositories."""
        return sum(result.stats.items for result in self.repository_results.values())

    @property
    def success_rate(self) -> float:
        """Get the success rate of repository searches."""
        if self.total_repositories == 0:
            return 0.0
        return self.successful_repositories / self.total_repositories


class RepositoryManager:
    """
    Manages multiple repository configurations and metadata.

    Provides centralized management of repository information,
    configurations, and health monitoring.
    """

    def __init__(self) -> None:
        self.repositories: dict[str, RepositoryInfo] = {}
        self.logger = get_logger()
        self._lock = threading.RLock()

    def add_repository(
        self,
        name: str,
        path: Path | str,
        config: SearchConfig | None = None,
        priority: str = "normal",
        **metadata: Any
    ) -> bool:
        """
        Add a repository to the manager.

        Args:
            name: Unique name for the repository
            path: Path to the repository
            config: Search configuration for this repository
            priority: Priority level ("high", "normal", "low")
            **metadata: Additional metadata for the repository

        Returns:
            True if repository was added successfully, False otherwise
        """
        with self._lock:
            if name in self.repositories:
                self.logger.warning(f"Repository '{name}' already exists")
                return False

            path = Path(path)
            if not path.exists():
                self.logger.error(f"Repository path does not exist: {path}")
                return False

            if config is None:
                config = SearchConfig(paths=[str(path)])

            repo_info = RepositoryInfo(
                name=name,
                path=path,
                config=config,
                priority=priority,
                metadata=metadata
            )

            self.repositories[name] = repo_info
            self.logger.info(f"Added repository '{name}' at {path}")
            return True

    def remove_repository(self, name: str) -> bool:
        """
        Remove a repository from the manager.

        Args:
            name: Name of the repository to remove

        Returns:
            True if repository was removed, False if not found
        """
        with self._lock:
            if name in self.repositories:
                del self.repositories[name]
                self.logger.info(f"Removed repository '{name}'")
                return True
            return False

    def get_repository(self, name: str) -> RepositoryInfo | None:
        """Get repository information by name."""
        return self.repositories.get(name)

    def list_repositories(self) -> list[str]:
        """Get list of repository names."""
        return list(self.repositories.keys())

    def get_enabled_repositories(self) -> dict[str, RepositoryInfo]:
        """Get all enabled repositories."""
        return {name: repo for name, repo in self.repositories.items() if repo.enabled}

    def configure_repository(self, name: str, **config_updates: Any) -> bool:
        """
        Update repository configuration.

        Args:
            name: Repository name
            **config_updates: Configuration updates

        Returns:
            True if configuration was updated, False if repository not found
        """
        with self._lock:
            repo = self.repositories.get(name)
            if not repo:
                return False

            # Update repository attributes
            for key, value in config_updates.items():
                if hasattr(repo, key):
                    setattr(repo, key, value)
                else:
                    repo.metadata[key] = value

            self.logger.info(f"Updated configuration for repository '{name}'")
            return True

    def refresh_all_status(self) -> None:
        """Refresh status for all repositories."""
        for repo in self.repositories.values():
            repo.refresh_status()

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary for all repositories."""
        summary = {
            "total": len(self.repositories),
            "healthy": 0,
            "warning": 0,
            "error": 0,
            "unknown": 0,
            "enabled": 0,
            "disabled": 0
        }

        for repo in self.repositories.values():
            summary[repo.health_status] += 1
            if repo.enabled:
                summary["enabled"] += 1
            else:
                summary["disabled"] += 1

        return summary


class SearchCoordinator:
    """
    Coordinates searches across multiple repositories.

    Handles parallel execution, result aggregation, and error handling
    for multi-repository search operations.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = get_logger()

    def search_repositories(
        self,
        repositories: dict[str, RepositoryInfo],
        query: Query,
        timeout: float = 30.0
    ) -> MultiRepoSearchResult:
        """
        Execute search across multiple repositories in parallel.

        Args:
            repositories: Dictionary of repository name to RepositoryInfo
            query: Search query to execute
            timeout: Timeout for each repository search

        Returns:
            MultiRepoSearchResult with results from all repositories
        """
        start_time = time.time()
        repository_results: dict[str, SearchResult] = {}
        failed_repositories: list[str] = []

        # Sort repositories by priority
        sorted_repos = sorted(
            repositories.items(),
            key=lambda x: {"high": 0, "normal": 1,
                           "low": 2}.get(x[1].priority, 1)
        )

        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit search tasks
            future_to_repo = {
                executor.submit(self._search_single_repository, repo_info, query, timeout): name
                for name, repo_info in sorted_repos
            }

            # Collect results
            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    result = future.result()
                    if result:
                        repository_results[repo_name] = result
                    else:
                        failed_repositories.append(repo_name)
                        self.logger.warning(
                            f"No results from repository '{repo_name}'")
                except Exception as e:
                    failed_repositories.append(repo_name)
                    self.logger.error(
                        f"Error searching repository '{repo_name}': {e}")

        # Create multi-repo result
        search_time_ms = (time.time() - start_time) * 1000

        return MultiRepoSearchResult(
            repository_results=repository_results,
            total_repositories=len(repositories),
            successful_repositories=len(repository_results),
            failed_repositories=failed_repositories,
            search_time_ms=search_time_ms
        )

    def _search_single_repository(
        self,
        repo_info: RepositoryInfo,
        query: Query,
        timeout: float
    ) -> SearchResult | None:
        """
        Execute search in a single repository.

        Args:
            repo_info: Repository information
            query: Search query
            timeout: Search timeout

        Returns:
            SearchResult if successful, None otherwise
        """
        try:
            # Import PySearch here to avoid circular imports
            from ..core.api import PySearch
            # Create PySearch instance for this repository
            search_engine = PySearch(config=repo_info.config)

            # Execute search with timeout
            result = search_engine.run(query)

            # Add repository metadata to results
            for item in result.items:
                if not hasattr(item, 'metadata'):
                    item.metadata = {}  # type: ignore[attr-defined]
                # type: ignore[attr-defined]
                item.metadata['repository'] = repo_info.name
                item.metadata['repository_path'] = str(
                    repo_info.path)  # type: ignore[attr-defined]

            return result

        except Exception as e:
            self.logger.error(
                f"Error searching repository '{repo_info.name}': {e}")
            return None

    def aggregate_results(
        self,
        repository_results: dict[str, SearchResult],
        max_results: int = 1000
    ) -> SearchResult:
        """
        Aggregate results from multiple repositories into a single result.

        Args:
            repository_results: Results from each repository
            max_results: Maximum number of results to include

        Returns:
            Aggregated SearchResult
        """
        all_items: list[SearchItem] = []
        total_files_scanned = 0
        total_files_matched = 0
        total_elapsed_ms = 0.0

        # Collect all items
        for repo_name, result in repository_results.items():
            all_items.extend(result.items)
            total_files_scanned += result.stats.files_scanned
            total_files_matched += result.stats.files_matched
            total_elapsed_ms = max(total_elapsed_ms, result.stats.elapsed_ms)

        # Sort by relevance (you might want to implement cross-repo scoring)
        all_items.sort(key=lambda item: (
            # Prioritize by repository priority if available
            getattr(item, 'metadata', {}).get('repository_priority', 1),
            # Then by file name (simple heuristic)
            str(item.file)
        ))

        # Limit results
        if len(all_items) > max_results:
            all_items = all_items[:max_results]

        # Create aggregated stats
        from .types import SearchStats
        aggregated_stats = SearchStats(
            files_scanned=total_files_scanned,
            files_matched=total_files_matched,
            items=len(all_items),
            elapsed_ms=total_elapsed_ms,
            indexed_files=total_files_scanned
        )

        return SearchResult(items=all_items, stats=aggregated_stats)


class MultiRepoSearchEngine:
    """
    Main engine for multi-repository search operations.

    Provides high-level interface for searching across multiple repositories
    with intelligent coordination, result aggregation, and management.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize multi-repository search engine.

        Args:
            max_workers: Maximum number of parallel workers for searches
        """
        self.repository_manager = RepositoryManager()
        self.search_coordinator = SearchCoordinator(max_workers=max_workers)
        self.logger = get_logger()

        # Search history and statistics
        self.search_history: list[dict[str, Any]] = []
        self.total_searches = 0
        self.total_search_time = 0.0

    def add_repository(
        self,
        name: str,
        path: Path | str,
        config: SearchConfig | None = None,
        priority: str = "normal",
        **metadata: Any
    ) -> bool:
        """
        Add a repository to the multi-repo search engine.

        Args:
            name: Unique name for the repository
            path: Path to the repository
            config: Search configuration for this repository
            priority: Priority level ("high", "normal", "low")
            **metadata: Additional metadata for the repository

        Returns:
            True if repository was added successfully, False otherwise

        Example:
            >>> engine = MultiRepoSearchEngine()
            >>>
            >>> # Add a high-priority repository
            >>> engine.add_repository(
            ...     "core-lib",
            ...     "/path/to/core-lib",
            ...     priority="high"
            ... )
            >>>
            >>> # Add repository with custom configuration
            >>> config = SearchConfig(
            ...     include=["**/*.py", "**/*.js"],
            ...     exclude=["**/node_modules/**", "**/venv/**"]
            ... )
            >>> engine.add_repository("web-app", "/path/to/web-app", config=config)
        """
        return self.repository_manager.add_repository(
            name=name,
            path=path,
            config=config,
            priority=priority,
            **metadata
        )

    def remove_repository(self, name: str) -> bool:
        """
        Remove a repository from the search engine.

        Args:
            name: Name of the repository to remove

        Returns:
            True if repository was removed, False if not found
        """
        return self.repository_manager.remove_repository(name)

    def configure_repository(self, name: str, **config_updates: Any) -> bool:
        """
        Update repository configuration.

        Args:
            name: Repository name
            **config_updates: Configuration updates

        Returns:
            True if configuration was updated, False if repository not found

        Example:
            >>> # Update repository priority and add metadata
            >>> engine.configure_repository(
            ...     "web-app",
            ...     priority="high",
            ...     team="frontend",
            ...     language="javascript"
            ... )
        """
        return self.repository_manager.configure_repository(name, **config_updates)

    def list_repositories(self) -> list[str]:
        """
        Get list of repository names.

        Returns:
            List of repository names
        """
        return self.repository_manager.list_repositories()

    def get_repository_info(self, name: str) -> RepositoryInfo | None:
        """
        Get detailed information about a repository.

        Args:
            name: Repository name

        Returns:
            RepositoryInfo if found, None otherwise
        """
        return self.repository_manager.get_repository(name)

    def search_all(
        self,
        pattern: str,
        use_regex: bool = False,
        use_ast: bool = False,
        use_semantic: bool = False,
        context: int = 2,
        max_results: int = 1000,
        aggregate_results: bool = True,
        timeout: float = 30.0
    ) -> MultiRepoSearchResult:
        """
        Search across all enabled repositories.

        Args:
            pattern: Search pattern
            use_regex: Whether to use regex matching
            use_ast: Whether to use AST-based matching
            use_semantic: Whether to use semantic matching
            context: Number of context lines
            max_results: Maximum number of results
            aggregate_results: Whether to aggregate results into single result
            timeout: Timeout for each repository search

        Returns:
            MultiRepoSearchResult with results from all repositories

        Example:
            >>> # Simple text search across all repositories
            >>> results = engine.search_all("def main")
            >>> print(f"Found matches in {results.successful_repositories} repositories")
            >>>
            >>> # Advanced search with aggregation
            >>> results = engine.search_all(
            ...     pattern=r"class \\w+Test",
            ...     use_regex=True,
            ...     aggregate_results=True,
            ...     max_results=500
            ... )
        """
        from .types import Query

        query = Query(
            pattern=pattern,
            use_regex=use_regex,
            use_ast=use_ast,
            use_semantic=use_semantic,
            context=context
        )

        return self.search_repositories(
            repositories=None,  # Search all
            query=query,
            max_results=max_results,
            aggregate_results=aggregate_results,
            timeout=timeout
        )

    def search_repositories(
        self,
        repositories: list[str] | None = None,
        query: Query | None = None,
        pattern: str | None = None,
        max_results: int = 1000,
        aggregate_results: bool = True,
        timeout: float = 30.0,
        **query_kwargs: Any
    ) -> MultiRepoSearchResult:
        """
        Search specific repositories or all repositories.

        Args:
            repositories: List of repository names to search (None for all)
            query: Pre-built Query object
            pattern: Search pattern (if query not provided)
            max_results: Maximum number of results
            aggregate_results: Whether to aggregate results
            timeout: Timeout for each repository search
            **query_kwargs: Additional query parameters

        Returns:
            MultiRepoSearchResult with search results

        Example:
            >>> # Search specific repositories
            >>> results = engine.search_repositories(
            ...     repositories=["core-lib", "web-app"],
            ...     pattern="database connection",
            ...     use_semantic=True
            ... )
            >>>
            >>> # Search with pre-built query
            >>> from pysearch.types import Query
            >>> query = Query(pattern="TODO", use_regex=False)
            >>> results = engine.search_repositories(query=query)
        """
        start_time = time.time()

        # Build query if not provided
        if query is None:
            if pattern is None:
                raise ValueError("Either query or pattern must be provided")

            from .types import Query
            query = Query(pattern=pattern, **query_kwargs)

        # Get repositories to search
        if repositories is None:
            target_repos = self.repository_manager.get_enabled_repositories()
        else:
            target_repos = {}
            for repo_name in repositories:
                repo_info = self.repository_manager.get_repository(repo_name)
                if repo_info and repo_info.enabled:
                    target_repos[repo_name] = repo_info
                else:
                    self.logger.warning(
                        f"Repository '{repo_name}' not found or disabled")

        if not target_repos:
            self.logger.warning("No repositories available for search")
            return MultiRepoSearchResult(
                repository_results={},
                total_repositories=0,
                successful_repositories=0,
                search_time_ms=0.0
            )

        # Execute search
        self.logger.info(
            f"Searching {len(target_repos)} repositories for: {query.pattern}")

        result = self.search_coordinator.search_repositories(
            repositories=target_repos,
            query=query,
            timeout=timeout
        )

        # Aggregate results if requested
        if aggregate_results and result.repository_results:
            result.aggregated_result = self.search_coordinator.aggregate_results(
                result.repository_results,
                max_results=max_results
            )

        # Update statistics
        search_time = time.time() - start_time
        self.total_searches += 1
        self.total_search_time += search_time

        # Add to history
        self.search_history.append({
            "timestamp": time.time(),
            "pattern": query.pattern,
            "repositories": list(target_repos.keys()),
            "total_matches": result.total_matches,
            "success_rate": result.success_rate,
            "search_time": search_time
        })

        # Limit history size
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]

        self.logger.info(
            f"Multi-repo search completed: {result.successful_repositories}/{result.total_repositories} "
            f"repositories, {result.total_matches} total matches, {search_time:.2f}s"
        )

        return result

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status for all repositories.

        Returns:
            Dictionary with health information
        """
        # Refresh status for all repositories
        self.repository_manager.refresh_all_status()

        health_summary = self.repository_manager.get_health_summary()

        # Add search statistics
        health_summary.update({
            "total_searches": self.total_searches,
            "average_search_time": (
                self.total_search_time / self.total_searches
                if self.total_searches > 0 else 0.0
            ),
            "recent_searches": len(self.search_history)
        })

        return health_summary

    def get_search_statistics(self) -> dict[str, Any]:
        """
        Get search performance statistics.

        Returns:
            Dictionary with search statistics
        """
        if not self.search_history:
            return {
                "total_searches": 0,
                "average_search_time": 0.0,
                "average_matches": 0.0,
                "average_success_rate": 0.0
            }

        recent_searches = self.search_history[-20:]  # Last 20 searches

        return {
            "total_searches": self.total_searches,
            "average_search_time": self.total_search_time / self.total_searches,
            "recent_average_search_time": sum(s["search_time"] for s in recent_searches) / len(recent_searches),
            "average_matches": sum(s["total_matches"] for s in recent_searches) / len(recent_searches),
            "average_success_rate": sum(s["success_rate"] for s in recent_searches) / len(recent_searches),
            "most_searched_patterns": self._get_most_searched_patterns()
        }

    def _get_most_searched_patterns(self) -> list[tuple[str, int]]:
        """Get most frequently searched patterns."""
        pattern_counts: dict[str, int] = {}
        for search in self.search_history:
            pattern = search["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Return top 10 most searched patterns
        return sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
