"""
Main API module for pysearch.

This module provides the primary PySearch class that serves as the main entry point
for programmatic access to the search engine. It coordinates all search operations
including indexing, matching, scoring, and result formatting.

Classes:
    PySearch: Main search engine class that orchestrates all search operations

Key Features:
    - File content caching with TTL for performance
    - Parallel search execution with configurable workers
    - Metadata filtering and author information extraction
    - Result deduplication and similarity clustering
    - Comprehensive error handling and logging
    - Search history tracking

Example:
    Basic search operation:
        >>> from pysearch.api import PySearch
        >>> from pysearch.config import SearchConfig
        >>> from pysearch.types import Query
        >>>
        >>> config = SearchConfig(paths=["."], include=["**/*.py"])
        >>> engine = PySearch(config)
        >>> query = Query(pattern="def main", use_regex=True)
        >>> results = engine.run(query)
        >>> print(f"Found {len(results.items)} matches in {results.stats.elapsed_ms}ms")

    Advanced search with filters:
        >>> from pysearch.types import ASTFilters
        >>> filters = ASTFilters(func_name="main", decorator="lru_cache")
        >>> query = Query(pattern="def", use_ast=True, ast_filters=filters)
        >>> results = engine.run(query)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..indexing.cache import CacheManager
from ..indexing.indexer import Indexer
from ..integrations.multi_repo import MultiRepoSearchEngine, MultiRepoSearchResult, RepositoryInfo
from ..search.boolean import evaluate_boolean_query_with_items, parse_boolean_query
from ..search.matchers import search_in_file
from ..search.scorer import (
    RankingStrategy,
    cluster_results_by_similarity,
    deduplicate_overlapping_results,
    sort_items,
)
from ..storage.qdrant_client import QdrantConfig
from ..utils.error_handling import ErrorCollector, create_error_report
from ..utils.file_watcher import FileEvent
from ..utils.logging_config import SearchLogger, get_logger
from ..utils.metadata_filters import apply_metadata_filters, get_file_author
from ..utils.helpers import create_file_metadata
from .config import SearchConfig
from .history import SearchHistory
from .history.history_core import SearchHistoryEntry
from .managers.hybrid_search import HybridSearchManager
from .managers.cache_integration import CacheIntegrationManager
from .managers.dependency_integration import DependencyIntegrationManager
from .managers.indexing_integration import IndexingIntegrationManager
from .managers.file_watching import FileWatchingManager
from .managers.graphrag_integration import GraphRAGIntegrationManager
from .managers.multi_repo_integration import MultiRepoIntegrationManager
from .managers.parallel_processing import ParallelSearchManager
from .types import (
    CountResult,
    GraphRAGQuery,
    OutputFormat,
    Query,
    SearchItem,
    SearchResult,
    SearchStats,
)


class PySearch:
    """
    Main search engine class for pysearch.

    This class orchestrates all search operations including file indexing, content
    matching, result scoring, and output formatting. It provides both high-level
    convenience methods and low-level control for advanced use cases.

    Attributes:
        cfg (SearchConfig): Configuration object controlling search behavior
        indexer (Indexer): File indexing and caching manager
        history (SearchHistory): Search history tracking
        logger (SearchLogger): Logging interface
        error_collector (ErrorCollector): Error collection and reporting
        cache_ttl (int): Time-to-live for in-memory caches in seconds

    Example:
        Basic usage:
            >>> from pysearch import PySearch, SearchConfig
            >>> from pysearch.types import Query
            >>>
            >>> config = SearchConfig(paths=["."], include=["**/*.py"])
            >>> engine = PySearch(config)
            >>>
            >>> # Simple text search
            >>> results = engine.search("def main")
            >>> print(f"Found {len(results.items)} matches")
            >>>
            >>> # Advanced query with filters
            >>> from pysearch.types import ASTFilters
            >>> filters = ASTFilters(func_name="main")
            >>> query = Query(pattern="def", use_ast=True, ast_filters=filters)
            >>> results = engine.run(query)

        With custom configuration:
            >>> config = SearchConfig(
            ...     paths=["./src", "./tests"],
            ...     context=5,
            ...     parallel=True,
            ...     workers=4
            ... )
            >>> engine = PySearch(config)
    """

    def __init__(
        self,
        config: SearchConfig | None = None,
        logger: SearchLogger | None = None,
        qdrant_config: QdrantConfig | None = None,
        enable_graphrag: bool = False,
        enable_metadata_indexing: bool = False,
    ) -> None:
        """
        Initialize the PySearch engine.

        Args:
            config: Search configuration object. If None, uses default configuration.
            logger: Custom logger instance. If None, uses default logger.
            qdrant_config: Qdrant vector database configuration for GraphRAG.
            enable_graphrag: Whether to enable GraphRAG capabilities.
            enable_metadata_indexing: Whether to enable metadata indexing.
        """
        # Core components
        self.cfg = config or SearchConfig()
        self.cfg.enable_graphrag = enable_graphrag
        self.cfg.enable_metadata_indexing = enable_metadata_indexing

        self.indexer = Indexer(self.cfg)
        self.history = SearchHistory(self.cfg)
        self.logger = logger or get_logger()
        self.error_collector = ErrorCollector()

        # Integration managers
        self.hybrid_search = HybridSearchManager(self.cfg)
        self.cache_integration = CacheIntegrationManager(self.cfg)
        self.dependency_integration = DependencyIntegrationManager(self.cfg)
        self.file_watching = FileWatchingManager(self.cfg)
        self.graphrag_integration = GraphRAGIntegrationManager(self.cfg, qdrant_config)
        self.indexing_integration = IndexingIntegrationManager(self.cfg)
        self.multi_repo_integration = MultiRepoIntegrationManager(self.cfg)
        self.parallel_processing = ParallelSearchManager(self.cfg)

        # Initialize state attributes
        self._caching_enabled = False
        self._multi_repo_enabled = False
        self.cache_manager: Any = None
        self.watch_manager: Any = None  # Will be set by file watching integration if needed
        self.multi_repo_engine: Any = None

        # Set up dependencies between managers
        self.hybrid_search.set_dependencies(self.error_collector, self.logger)
        self.graphrag_integration.set_dependencies(self.logger, self.error_collector)
        self.indexing_integration.set_dependencies(self.logger, self.error_collector)
        self.dependency_integration.set_logger(self.logger)
        self.file_watching.set_indexer(self.indexer)

    async def initialize_graphrag(self) -> None:
        """Initialize GraphRAG components."""
        await self.graphrag_integration.initialize()

    async def initialize_metadata_indexing(self) -> None:
        """Initialize metadata indexing components."""
        await self.indexing_integration.initialize()

    async def build_knowledge_graph(self, force_rebuild: bool = False) -> bool:
        """Build the GraphRAG knowledge graph."""
        return await self.graphrag_integration.build_knowledge_graph(force_rebuild)

    async def build_metadata_index(
        self, include_semantic: bool = True, force_rebuild: bool = False
    ) -> bool:
        """Build the metadata index."""
        return await self.indexing_integration.build_index(force_rebuild)

    async def graphrag_search(self, query: GraphRAGQuery) -> Any | None:
        """Perform GraphRAG-based search."""
        return await self.graphrag_integration.query_graph(query)

    async def metadata_index_search(self, query: Any) -> dict[str, Any] | None:
        """Search using the metadata index."""
        return await self.indexing_integration.query_index(query)

    def _search_file(self, path: Path, query: Query) -> list[SearchItem]:
        text = self._get_cached_file_content(path)
        if text is None:
            return []

        # Apply metadata filters if specified
        if query.metadata_filters is not None:
            metadata = create_file_metadata(path, text)
            if metadata is None:
                return []

            # Add author information if needed
            if query.metadata_filters.author_pattern is not None and metadata.author is None:
                metadata.author = get_file_author(path)

            if not apply_metadata_filters(metadata, query.metadata_filters):
                return []

        # Handle boolean queries
        if query.use_boolean:
            try:
                boolean_query = parse_boolean_query(query.pattern)
                # Get basic matches first, then filter with boolean logic
                basic_items = search_in_file(path, text, query)
                items = evaluate_boolean_query_with_items(boolean_query, text, basic_items)
            except Exception as e:
                self.logger.warning(f"Boolean query error in {path}: {e}")
                return []
        else:
            items = search_in_file(path, text, query)

        # Apply per-file limit if specified
        if query.max_per_file is not None and len(items) > query.max_per_file:
            items = items[: query.max_per_file]

        return items

    def _get_cached_file_content(self, path: Path) -> str:
        """
        Get file content with caching based on modification time.

        This method delegates to the cache integration manager for file content caching.

        Args:
            path: Path to the file to read

        Returns:
            File content as string if successful, empty string if file cannot be read
        """
        content = self.cache_integration.get_cached_file_content(path)
        return content if content is not None else ""

    def _get_cache_key(self, query: Query) -> str:
        """Generate cache key for search query."""
        return self.cache_integration._generate_cache_key(query)

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return self.cache_integration._is_cache_valid(timestamp)

    def run(self, query: Query, use_cache: bool = True) -> SearchResult:
        r"""
        Execute a search query and return results.

        This is the main search method that coordinates all search operations:
        file indexing, content matching, result scoring, and caching.

        Args:
            query: Query object specifying search parameters including pattern,
                  search modes (regex, AST, semantic), filters, and output format

        Returns:
            SearchResult containing matched items and performance statistics

        Example:
            >>> from pysearch.types import Query, ASTFilters
            >>>
            >>> # Simple text search
            >>> query = Query(pattern="def main")
            >>> results = engine.run(query)
            >>>
            >>> # AST search with filters
            >>> filters = ASTFilters(func_name="main")
            >>> query = Query(pattern="def", use_ast=True, ast_filters=filters)
            >>> results = engine.run(query)
            >>>
            >>> # Regex search
            >>> query = Query(pattern=r"def \\w+_handler", use_regex=True)
            >>> results = engine.run(query)
        """
        # Clear previous errors
        self.error_collector.clear()

        # Log search start
        self.logger.log_search_start(
            pattern=query.pattern,
            paths=self.cfg.paths,
            use_regex=query.use_regex,
            use_ast=query.use_ast,
        )

        # Check cache first if enabled
        if use_cache:
            cached_result = self.cache_integration.get_cached_result(query)
            if cached_result:
                self.logger.debug(f"Using cached result for query: {query.pattern}")
                return cached_result

        t0 = time.perf_counter()

        try:
            changed, removed, total_seen = self.indexer.scan()
            self.indexer.save()
            self.logger.debug(
                f"Indexer scan: {total_seen} files seen, {len(changed or [])} changed"
            )
        except Exception as e:
            self.logger.error(f"Error during indexing: {e}")
            self.error_collector.add_error(e)
            # Continue with empty file list
            changed = []
            total_seen = 0

        paths = changed or list(self.indexer.iter_all_paths())
        self.logger.debug(f"Searching in {len(paths)} files")

        # Execute search using parallel processing manager
        try:
            items = self.parallel_processing.search_files(paths, query, self._search_file)
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            self.error_collector.add_error(e)
            items = []

        # Sort results by relevance with configurable strategy
        try:
            # Use hybrid ranking by default, could be configurable
            items = sort_items(items, self.cfg, query.pattern, RankingStrategy.HYBRID)
            # Remove overlapping results for cleaner output
            items = deduplicate_overlapping_results(items)
        except Exception as e:
            self.logger.error(f"Error sorting/deduplicating results: {e}")
            self.error_collector.add_error(e)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Log search completion
        self.logger.log_search_complete(
            pattern=query.pattern,
            results_count=len(items),
            elapsed_ms=elapsed_ms,
            files_scanned=total_seen,
        )

        stats = SearchStats(
            files_scanned=total_seen,
            files_matched=len({it.file for it in items}),
            items=len(items),
            elapsed_ms=elapsed_ms,
            indexed_files=self.indexer.count_indexed(),
        )

        result = SearchResult(items=items, stats=stats)

        # Add to search history
        try:
            self.history.add_search(query, result)
        except Exception as e:
            self.logger.warning(f"Could not save to history: {e}")

        # Cache result if enabled
        if use_cache:
            self.cache_integration.cache_result(query, result)

        return result

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self.cache_integration.clear_caches()

    # Convenience text/regex api
    def search(
        self,
        pattern: str,
        regex: bool = False,
        context: int | None = None,
        output: OutputFormat = OutputFormat.TEXT,
        **kwargs: Any,
    ) -> SearchResult:
        r"""
        Convenience method for simple text or regex searches.

        This method provides a simplified interface for common search operations
        without requiring explicit Query object construction.

        Args:
            pattern: Search pattern (text or regex)
            regex: Whether to treat pattern as regex (default: False)
            context: Number of context lines around matches (overrides config)
            output: Output format for results
            **kwargs: Additional parameters:
                - use_ast: Enable AST-based search
                - filters: AST filters for structural matching
                - metadata_filters: Metadata-based filters

        Returns:
            SearchResult containing matched items and statistics

        Example:
            >>> # Simple text search
            >>> results = engine.search("def main")
            >>>
            >>> # Regex search with context
            >>> results = engine.search(r"def \\w+_handler", regex=True, context=5)
            >>>
            >>> # AST search with filters
            >>> from pysearch.types import ASTFilters
            >>> filters = ASTFilters(func_name="main")
            >>> results = engine.search("def", use_ast=True, filters=filters)
        """
        q = Query(
            pattern=pattern,
            use_regex=regex,
            use_ast=kwargs.get("use_ast", False),
            context=context if context is not None else self.cfg.context,
            output=output,
            filters=kwargs.get("filters"),
            metadata_filters=kwargs.get("metadata_filters"),
            search_docstrings=self.cfg.enable_docstrings,
            search_comments=self.cfg.enable_comments,
            search_strings=self.cfg.enable_strings,
        )
        return self.run(q)

    def search_count_only(
        self,
        pattern: str,
        regex: bool = False,
        use_boolean: bool = False,
        **kwargs: Any,
    ) -> CountResult:
        r"""
        Perform a count-only search that returns only match counts, not content.

        This method is optimized for fast counting and doesn't return actual
        match content, making it suitable for quick exploration or validation.

        Args:
            pattern: Search pattern (text, regex, or boolean query)
            regex: Whether to treat pattern as regex (default: False)
            use_boolean: Whether to use boolean query logic (default: False)
            **kwargs: Additional parameters for filtering

        Returns:
            CountResult containing match counts and statistics

        Example:
            >>> # Count function definitions
            >>> result = engine.search_count_only("def ")
            >>> print(f"Found {result.total_matches} function definitions")
            >>>
            >>> # Count with boolean logic
            >>> result = engine.search_count_only("(async AND handler) NOT test", use_boolean=True)
            >>> print(f"Found {result.total_matches} matches in {result.files_matched} files")
        """
        # Create query with count_only flag
        query = Query(
            pattern=pattern,
            use_regex=regex,
            use_boolean=use_boolean,
            count_only=True,
            context=0,  # No context needed for counting
            filters=kwargs.get("filters"),
            metadata_filters=kwargs.get("metadata_filters"),
            search_docstrings=self.cfg.enable_docstrings,
            search_comments=self.cfg.enable_comments,
            search_strings=self.cfg.enable_strings,
        )

        # Run search to get items
        result = self.run(query)

        # Convert to count result
        return CountResult(
            total_matches=len(result.items),
            files_matched=result.stats.files_matched,
            stats=result.stats,
        )

    async def hybrid_search(
        self,
        pattern: str,
        use_graphrag: bool = True,
        use_metadata_index: bool = True,
        graphrag_max_hops: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform hybrid search combining traditional search, GraphRAG, and metadata indexing.

        This method provides a unified interface that leverages all available search
        capabilities to provide comprehensive results.

        Args:
            pattern: Search pattern or query
            use_graphrag: Whether to include GraphRAG results
            use_metadata_index: Whether to include metadata index results
            graphrag_max_hops: Maximum hops for GraphRAG traversal
            **kwargs: Additional parameters for traditional search

        Returns:
            Dictionary containing results from all enabled search methods
        """
        return await self.hybrid_search.hybrid_search(
            pattern=pattern,
            traditional_search_func=self.run,
            graphrag_search_func=self.graphrag_search if use_graphrag else None,
            metadata_index_search_func=self.metadata_index_search if use_metadata_index else None,
            use_graphrag=use_graphrag,
            use_metadata_index=use_metadata_index,
            graphrag_max_hops=graphrag_max_hops,
            **kwargs,
        )

    async def close_async_components(self) -> None:
        """Close async components properly."""
        await self.graphrag_integration.close()
        await self.indexing_integration.close()

    def search_semantic(
        self, query: str, threshold: float = 0.1, max_results: int = 100, **kwargs: Any
    ) -> SearchResult:
        """
        Perform semantic search with embedding-based similarity.

        This method uses sophisticated semantic analysis including:
        - Vector-based similarity using TF-IDF embeddings
        - Code structure awareness (functions, classes, imports)
        - Contextual understanding and concept extraction
        - Multi-modal scoring combining multiple similarity metrics

        Args:
            query: Semantic search query (e.g., "database connection", "web api")
            threshold: Minimum semantic similarity threshold (0.0-1.0)
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            SearchResult with semantically relevant matches

        Example:
            >>> # Find database-related code
            >>> results = engine.search_semantic("database connection")
            >>>
            >>> # Find web API implementations
            >>> results = engine.search_semantic("web api", threshold=0.2)
            >>>
            >>> # Find testing utilities
            >>> results = engine.search_semantic("test utilities", max_results=50)
        """
        # Clear previous errors
        self.error_collector.clear()

        # Log search start
        self.logger.log_search_start(
            pattern=f"semantic:{query}",
            paths=self.cfg.paths,
            use_regex=False,
            use_ast=False,
        )

        # Get files to search
        try:
            changed, removed, total_seen = self.indexer.scan()
            self.logger.debug(
                f"Indexer scan: {len(changed)} changed, {len(removed)} removed, {total_seen} total"
            )
        except Exception as e:
            self.error_collector.add_error(e)
            # Continue with empty file list
            changed, removed, total_seen = [], [], 0

        paths = changed or list(self.indexer.iter_all_paths())

        # Delegate to hybrid search manager
        return self.hybrid_search.search_semantic(
            query=query,
            file_paths=paths,
            get_file_content_func=self._get_cached_file_content,
            threshold=threshold,
            max_results=max_results,
            **kwargs,
        )

    def analyze_dependencies(self, directory: Path | None = None, recursive: bool = True) -> Any:
        """
        Analyze code dependencies and build a dependency graph.

        This method performs comprehensive dependency analysis including:
        - Import statement extraction across multiple languages
        - Dependency graph construction
        - Circular dependency detection
        - Module coupling analysis

        Args:
            directory: Directory to analyze (defaults to first configured path)
            recursive: Whether to analyze subdirectories recursively

        Returns:
            DependencyGraph with complete dependency information

        Example:
            >>> # Analyze current project dependencies
            >>> graph = engine.analyze_dependencies()
            >>> print(f"Found {len(graph.nodes)} modules")
            >>>
            >>> # Check for circular dependencies
            >>> from pysearch.dependency_analysis import CircularDependencyDetector
            >>> detector = CircularDependencyDetector(graph)
            >>> cycles = detector.find_cycles()
            >>> if cycles:
            ...     print(f"Warning: {len(cycles)} circular dependencies found")
        """
        return self.dependency_integration.analyze_dependencies(directory, recursive)

    def get_dependency_metrics(self, graph: Any | None = None) -> Any:
        """
        Calculate comprehensive dependency metrics.

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            DependencyMetrics with detailed analysis results

        Example:
            >>> metrics = engine.get_dependency_metrics()
            >>> print(f"Total modules: {metrics.total_modules}")
            >>> print(f"Circular dependencies: {metrics.circular_dependencies}")
            >>> print(f"Average dependencies per module: {metrics.average_dependencies_per_module:.2f}")
            >>>
            >>> if metrics.highly_coupled_modules:
            ...     print("Highly coupled modules:")
            ...     for module in metrics.highly_coupled_modules:
            ...         print(f"  - {module}")
        """
        return self.dependency_integration.get_dependency_metrics(graph)

    def find_dependency_impact(self, module: str, graph: Any | None = None) -> dict[str, Any]:
        """
        Analyze the impact of changes to a specific module.

        This method identifies all modules that would be affected by changes
        to the specified module, helping with impact analysis for refactoring.

        Args:
            module: Module name to analyze
            graph: Dependency graph to use (if None, analyzes current project)

        Returns:
            Dictionary with impact analysis results

        Example:
            >>> # Analyze impact of changing a core module
            >>> impact = engine.find_dependency_impact("src.core.database")
            >>> print(f"Modules affected: {impact['total_affected_modules']}")
            >>> print(f"Impact score: {impact['impact_score']:.2f}")
            >>>
            >>> print("Direct dependents:")
            >>> for dep in impact['direct_dependents']:
            ...     print(f"  - {dep}")
        """
        return self.dependency_integration.find_dependency_impact(module, graph)

    def suggest_refactoring_opportunities(self, graph: Any | None = None) -> list[dict[str, Any]]:
        """
        Suggest refactoring opportunities based on dependency analysis.

        Analyzes the dependency graph to identify potential improvements:
        - Circular dependencies to break
        - Highly coupled modules to split
        - Dead code to remove
        - Architecture improvements

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            List of refactoring suggestions with priorities and rationale

        Example:
            >>> suggestions = engine.suggest_refactoring_opportunities()
            >>> for suggestion in suggestions:
            ...     print(f"[{suggestion['priority'].upper()}] {suggestion['type']}")
            ...     print(f"  {suggestion['description']}")
            ...     print(f"  Rationale: {suggestion['rationale']}")
            ...     print()
        """
        return self.dependency_integration.suggest_refactoring_opportunities(graph)

    def enable_auto_watch(
        self, debounce_delay: float = 0.5, batch_size: int = 50, max_batch_delay: float = 5.0
    ) -> bool:
        """
        Enable automatic file watching for real-time index updates.

        When enabled, the search index will be automatically updated when files
        change in the configured search paths. This provides real-time search
        results without manual index refreshing.

        Args:
            debounce_delay: Delay in seconds before processing file changes
            batch_size: Maximum number of changes to batch together
            max_batch_delay: Maximum delay before processing a batch

        Returns:
            True if auto-watch was enabled successfully, False otherwise

        Example:
            >>> # Enable auto-watch with default settings
            >>> success = engine.enable_auto_watch()
            >>> if success:
            ...     print("Auto-watch enabled - search index will update automatically")
            >>>
            >>> # Enable with custom settings for high-frequency changes
            >>> engine.enable_auto_watch(debounce_delay=1.0, batch_size=100)
        """
        return self.file_watching.enable_auto_watch(debounce_delay, batch_size, max_batch_delay)

    def disable_auto_watch(self) -> None:
        """
        Disable automatic file watching.

        Stops all file watchers and disables real-time index updates.
        The search index will need to be manually refreshed after this.

        Example:
            >>> engine.disable_auto_watch()
            >>> print("Auto-watch disabled - manual index refresh required")
        """
        self.file_watching.disable_auto_watch()

    def is_auto_watch_enabled(self) -> bool:
        """
        Check if automatic file watching is enabled.

        Returns:
            True if auto-watch is enabled, False otherwise
        """
        return self.file_watching.is_auto_watch_enabled()

    def get_watch_stats(self) -> dict[str, Any]:
        """
        Get file watching statistics.

        Returns:
            Dictionary with detailed watching statistics for all watchers

        Example:
            >>> stats = engine.get_watch_stats()
            >>> print(f"Watchers active: {len(stats)}")
            >>> for name, watcher_stats in stats.items():
            ...     print(f"  {name}: {watcher_stats['events_processed']} events processed")
        """
        return self.file_watching.get_watch_stats()

    def add_custom_watcher(
        self,
        name: str,
        path: Path | str,
        change_handler: Callable[[list[FileEvent]], None],
        **kwargs: Any,
    ) -> bool:
        """
        Add a custom file watcher with a specific change handler.

        This allows for custom processing of file changes beyond the standard
        index updates. Useful for implementing custom workflows or integrations.

        Args:
            name: Unique name for the watcher
            path: Path to watch for changes
            change_handler: Function to call when files change
            **kwargs: Additional arguments for the FileWatcher

        Returns:
            True if watcher was added successfully, False otherwise

        Example:
            >>> def my_handler(events):
            ...     for event in events:
            ...         print(f"File {event.event_type.value}: {event.path}")
            >>>
            >>> success = engine.add_custom_watcher(
            ...     "my_watcher",
            ...     "/path/to/watch",
            ...     my_handler
            ... )
        """
        return self.file_watching.add_custom_watcher(
            name=name, path=path, change_handler=change_handler, **kwargs
        )

    def remove_watcher(self, name: str) -> bool:
        """
        Remove a file watcher by name.

        Args:
            name: Name of the watcher to remove

        Returns:
            True if watcher was removed successfully, False otherwise
        """
        return self.file_watching.remove_watcher(name)

    def list_watchers(self) -> list[str]:
        """
        Get list of active file watcher names.

        Returns:
            List of watcher names
        """
        return self.file_watching.list_watchers()

    def enable_caching(
        self,
        backend: str = "memory",
        cache_dir: Path | str | None = None,
        max_size: int = 1000,
        default_ttl: float = 3600,
        compression: bool = False,
    ) -> bool:
        """
        Enable search result caching for improved performance.

        Caching stores search results to avoid re-executing expensive searches.
        Results are automatically invalidated when files change (if file watching
        is enabled) or when they expire.

        Args:
            backend: Cache backend ("memory" or "disk")
            cache_dir: Directory for disk cache (required for disk backend)
            max_size: Maximum number of cached results
            default_ttl: Default cache time-to-live in seconds
            compression: Enable compression for cached results

        Returns:
            True if caching was enabled successfully, False otherwise

        Example:
            >>> # Enable memory caching with default settings
            >>> success = engine.enable_caching()
            >>> if success:
            ...     print("Caching enabled - searches will be faster")
            >>>
            >>> # Enable persistent disk caching
            >>> engine.enable_caching(
            ...     backend="disk",
            ...     cache_dir="/tmp/pysearch_cache",
            ...     max_size=5000,
            ...     default_ttl=7200  # 2 hours
            ... )
        """
        if self._caching_enabled:
            self.logger.warning("Caching already enabled")
            return True

        try:
            self.cache_manager = CacheManager(
                backend=backend,
                cache_dir=cache_dir,
                max_size=max_size,
                default_ttl=default_ttl,
                compression=compression,
            )

            self._caching_enabled = True
            self.logger.info(f"Caching enabled with {backend} backend")
            return True

        except Exception as e:
            self.logger.error(f"Failed to enable caching: {e}")
            return False

    def disable_caching(self) -> None:
        """
        Disable search result caching.

        Stops caching new results and optionally clears existing cache.

        Example:
            >>> engine.disable_caching()
            >>> print("Caching disabled")
        """
        if not self._caching_enabled or not self.cache_manager:
            return

        try:
            self.cache_manager.shutdown()
            self.cache_manager = None
            self._caching_enabled = False
            self.logger.info("Caching disabled")
        except Exception as e:
            self.logger.error(f"Error disabling caching: {e}")

    def is_caching_enabled(self) -> bool:
        """
        Check if search result caching is enabled.

        Returns:
            True if caching is enabled, False otherwise
        """
        return self._caching_enabled

    def clear_cache(self) -> None:
        """
        Clear all cached search results.

        Example:
            >>> engine.clear_cache()
            >>> print("Cache cleared")
        """
        if self.cache_manager:
            self.cache_manager.clear()
            self.logger.info("Search cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics, empty if caching disabled

        Example:
            >>> stats = engine.get_cache_stats()
            >>> if stats:
            ...     print(f"Cache hit rate: {stats['hit_rate']:.2%}")
            ...     print(f"Total entries: {stats['total_entries']}")
        """
        if self.cache_manager:
            return self.cache_manager.get_stats()  # type: ignore[no-any-return]
        return {}

    def invalidate_cache_for_file(self, file_path: Path | str) -> int:
        """
        Invalidate cached results that depend on a specific file.

        This is automatically called when file watching detects changes,
        but can be manually triggered when needed.

        Args:
            file_path: Path of the file that changed

        Returns:
            Number of cache entries invalidated

        Example:
            >>> # Manually invalidate cache for a changed file
            >>> count = engine.invalidate_cache_for_file("src/main.py")
            >>> print(f"Invalidated {count} cached results")
        """
        if self.cache_manager:
            return self.cache_manager.invalidate_by_file(str(file_path))  # type: ignore[no-any-return]
        return 0

    def _generate_cache_key(self, query: Query) -> str:
        """Generate a cache key for a search query."""
        return self.cache_integration._generate_cache_key(query)

    def _get_file_dependencies(self, result: SearchResult) -> set[str]:
        """Extract file dependencies from search result."""
        return {str(p) for p in self.cache_integration._get_file_dependencies(result)}

    def enable_multi_repo(self, max_workers: int = 4) -> bool:
        """
        Enable multi-repository search capabilities.

        This allows searching across multiple repositories simultaneously
        with intelligent coordination and result aggregation.

        Args:
            max_workers: Maximum number of parallel workers for searches

        Returns:
            True if multi-repo was enabled successfully, False otherwise

        Example:
            >>> # Enable multi-repository search
            >>> success = engine.enable_multi_repo()
            >>> if success:
            ...     print("Multi-repository search enabled")
            >>>
            >>> # Add repositories
            >>> engine.add_repository("project-a", "/path/to/project-a")
            >>> engine.add_repository("project-b", "/path/to/project-b")
        """
        if self._multi_repo_enabled:
            self.logger.warning("Multi-repository search already enabled")
            return True

        try:
            self.multi_repo_engine = MultiRepoSearchEngine(max_workers=max_workers)
            self._multi_repo_enabled = True
            self.logger.info("Multi-repository search enabled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to enable multi-repository search: {e}")
            return False

    def disable_multi_repo(self) -> None:
        """
        Disable multi-repository search capabilities.

        Example:
            >>> engine.disable_multi_repo()
            >>> print("Multi-repository search disabled")
        """
        if not self._multi_repo_enabled:
            return

        self.multi_repo_engine = None
        self._multi_repo_enabled = False
        self.logger.info("Multi-repository search disabled")

    def is_multi_repo_enabled(self) -> bool:
        """
        Check if multi-repository search is enabled.

        Returns:
            True if multi-repo is enabled, False otherwise
        """
        return self._multi_repo_enabled

    def add_repository(
        self,
        name: str,
        path: Path | str,
        config: SearchConfig | None = None,
        priority: str = "normal",
        **metadata: Any,
    ) -> bool:
        """
        Add a repository to multi-repository search.

        Args:
            name: Unique name for the repository
            path: Path to the repository
            config: Search configuration for this repository
            priority: Priority level ("high", "normal", "low")
            **metadata: Additional metadata for the repository

        Returns:
            True if repository was added successfully, False otherwise

        Example:
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
            ...     exclude=["**/node_modules/**"]
            ... )
            >>> engine.add_repository("web-app", "/path/to/web-app", config=config)
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            self.logger.error("Multi-repository search not enabled")
            return False

        return self.multi_repo_engine.add_repository(  # type: ignore[no-any-return]
            name=name, path=path, config=config, priority=priority, **metadata
        )

    def remove_repository(self, name: str) -> bool:
        """
        Remove a repository from multi-repository search.

        Args:
            name: Name of the repository to remove

        Returns:
            True if repository was removed, False if not found
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return False

        return self.multi_repo_engine.remove_repository(name)  # type: ignore[no-any-return]

    def list_repositories(self) -> list[str]:
        """
        Get list of repository names in multi-repository search.

        Returns:
            List of repository names, empty if multi-repo not enabled
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return []

        return self.multi_repo_engine.list_repositories()  # type: ignore[no-any-return]

    def get_repository_info(self, name: str) -> RepositoryInfo | None:
        """
        Get detailed information about a repository.

        Args:
            name: Repository name

        Returns:
            RepositoryInfo if found, None otherwise
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return None

        return self.multi_repo_engine.get_repository_info(name)  # type: ignore[no-any-return]

    def search_all_repositories(
        self,
        pattern: str,
        use_regex: bool = False,
        use_ast: bool = False,
        use_semantic: bool = False,
        context: int = 2,
        max_results: int = 1000,
        aggregate_results: bool = True,
        timeout: float = 30.0,
    ) -> MultiRepoSearchResult | None:
        """
        Search across all repositories in the multi-repository system.

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
            MultiRepoSearchResult with results from all repositories, None if not enabled

        Example:
            >>> # Simple text search across all repositories
            >>> results = engine.search_all_repositories("def main")
            >>> if results:
            ...     print(f"Found matches in {results.successful_repositories} repositories")
            >>>
            >>> # Advanced search with aggregation
            >>> results = engine.search_all_repositories(
            ...     pattern=r"class \\w+Test",
            ...     use_regex=True,
            ...     aggregate_results=True,
            ...     max_results=500
            ... )
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            self.logger.error("Multi-repository search not enabled")
            return None

        return self.multi_repo_engine.search_all(  # type: ignore[no-any-return]
            pattern=pattern,
            use_regex=use_regex,
            use_ast=use_ast,
            use_semantic=use_semantic,
            context=context,
            max_results=max_results,
            aggregate_results=aggregate_results,
            timeout=timeout,
        )

    def search_specific_repositories(
        self,
        repositories: list[str],
        query: Query,
        max_results: int = 1000,
        aggregate_results: bool = True,
        timeout: float = 30.0,
    ) -> MultiRepoSearchResult | None:
        """
        Search specific repositories with a pre-built query.

        Args:
            repositories: List of repository names to search
            query: Pre-built Query object
            max_results: Maximum number of results
            aggregate_results: Whether to aggregate results
            timeout: Timeout for each repository search

        Returns:
            MultiRepoSearchResult with search results, None if not enabled

        Example:
            >>> # Search specific repositories
            >>> from pysearch.types import Query
            >>> query = Query(pattern="TODO", use_regex=False)
            >>> results = engine.search_specific_repositories(
            ...     repositories=["core-lib", "web-app"],
            ...     query=query
            ... )
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            self.logger.error("Multi-repository search not enabled")
            return None

        return self.multi_repo_engine.search_repositories(  # type: ignore[no-any-return]
            repositories=repositories,
            query=query,
            max_results=max_results,
            aggregate_results=aggregate_results,
            timeout=timeout,
        )

    def get_multi_repo_health(self) -> dict[str, Any]:
        """
        Get health status for all repositories in multi-repository system.

        Returns:
            Dictionary with health information, empty if not enabled
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return {}

        return self.multi_repo_engine.get_health_status()  # type: ignore[no-any-return]

    def get_multi_repo_stats(self) -> dict[str, Any]:
        """
        Get search performance statistics for multi-repository system.

        Returns:
            Dictionary with search statistics, empty if not enabled
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return {}

        return self.multi_repo_engine.get_search_statistics()  # type: ignore[no-any-return]

    # New advanced search methods
    def fuzzy_search(
        self,
        pattern: str,
        max_distance: int = 2,
        min_similarity: float = 0.6,
        algorithm: str | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """
        Enhanced fuzzy search with multiple algorithms.

        Args:
            pattern: Search pattern
            max_distance: Maximum edit distance
            min_similarity: Minimum similarity score (0.0 to 1.0)
            algorithm: Fuzzy algorithm ('levenshtein', 'damerau_levenshtein', 'jaro_winkler', 'soundex', 'metaphone')
            **kwargs: Additional search parameters
        """
        from ..search.fuzzy import FuzzyAlgorithm, fuzzy_pattern

        # Parse algorithm parameter
        if algorithm:
            try:
                fuzzy_algo = FuzzyAlgorithm(algorithm.lower())
            except ValueError:
                fuzzy_algo = FuzzyAlgorithm.LEVENSHTEIN
        else:
            fuzzy_algo = FuzzyAlgorithm.LEVENSHTEIN

        fuzzy_regex = fuzzy_pattern(pattern, max_distance, fuzzy_algo)
        return self.search(fuzzy_regex, regex=True, **kwargs)

    def multi_algorithm_fuzzy_search(
        self,
        pattern: str,
        algorithms: list[str] | None = None,
        max_distance: int = 2,
        min_similarity: float = 0.6,
        **kwargs: Any,
    ) -> SearchResult:
        """
        Fuzzy search using multiple algorithms and combining results.

        Args:
            pattern: Search pattern
            algorithms: List of algorithm names to use
            max_distance: Maximum edit distance
            min_similarity: Minimum similarity score
            **kwargs: Additional search parameters
        """
        from ..search.fuzzy import FuzzyAlgorithm, fuzzy_pattern

        # Parse algorithms
        if algorithms:
            fuzzy_algos = []
            for algo_name in algorithms:
                try:
                    fuzzy_algos.append(FuzzyAlgorithm(algo_name.lower()))
                except ValueError:
                    continue
        else:
            fuzzy_algos = [
                FuzzyAlgorithm.LEVENSHTEIN,
                FuzzyAlgorithm.DAMERAU_LEVENSHTEIN,
                FuzzyAlgorithm.JARO_WINKLER,
            ]

        if not fuzzy_algos:
            return self.search(pattern, regex=False, **kwargs)

        # Run search with each algorithm's regex and merge results
        all_items: list[SearchItem] = []
        seen_keys: set[tuple[str, int]] = set()

        for algo in fuzzy_algos:
            try:
                fuzzy_regex = fuzzy_pattern(pattern, max_distance, algo)
                result = self.search(fuzzy_regex, regex=True, **kwargs)
                for item in result.items:
                    key = (str(item.file), item.start_line)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_items.append(item)
            except Exception:
                continue

        # Re-sort and deduplicate combined results
        from ..search.scorer import deduplicate_overlapping_results

        all_items = sort_items(all_items, self.cfg, pattern)
        all_items = deduplicate_overlapping_results(all_items)

        return SearchResult(
            items=all_items,
            stats=SearchStats(
                files_scanned=0,
                files_matched=len({it.file for it in all_items}),
                items=len(all_items),
                elapsed_ms=0.0,
                indexed_files=self.indexer.count_indexed(),
            ),
        )

    def phonetic_search(
        self, pattern: str, algorithm: str = "soundex", **kwargs: Any
    ) -> SearchResult:
        """
        Phonetic search for words that sound similar.

        Args:
            pattern: Search pattern
            algorithm: Phonetic algorithm ('soundex' or 'metaphone')
            **kwargs: Additional search parameters
        """
        from ..search.fuzzy import FuzzyAlgorithm, fuzzy_pattern

        if algorithm.lower() == "soundex":
            fuzzy_algo = FuzzyAlgorithm.SOUNDEX
        elif algorithm.lower() == "metaphone":
            fuzzy_algo = FuzzyAlgorithm.METAPHONE
        else:
            fuzzy_algo = FuzzyAlgorithm.SOUNDEX

        # Distance 0 for phonetic
        fuzzy_regex = fuzzy_pattern(pattern, 0, fuzzy_algo)
        return self.search(fuzzy_regex, regex=True, **kwargs)

    def semantic_search(self, concept: str, **kwargs: Any) -> SearchResult:
        """Semantic search based on code concepts."""
        from ..search.semantic import concept_to_patterns

        patterns = concept_to_patterns(concept)
        all_items = []
        for pattern in patterns:
            result = self.search(pattern, regex=True, **kwargs)
            all_items.extend(result.items)

        # Deduplicate and re-sort
        unique_items = list({(item.file, item.start_line): item for item in all_items}.values())
        unique_items = sort_items(unique_items, self.cfg, concept)

        return SearchResult(
            items=unique_items,
            stats=SearchStats(
                files_scanned=0,
                files_matched=len({it.file for it in unique_items}),
                items=len(unique_items),
                elapsed_ms=0.0,
                indexed_files=self.indexer.count_indexed(),
            ),
        )

    # History and bookmark management
    def get_search_history(self, limit: int | None = None) -> list[Any]:
        """Get search history entries."""
        return self.history.get_history(limit)

    def get_bookmarks(self) -> dict[str, SearchHistoryEntry]:
        """Get all bookmarks."""
        return self.history.get_bookmarks()

    def add_bookmark(self, name: str, query: Query, result: SearchResult) -> None:
        """Bookmark a search query and result."""
        self.history.add_bookmark(name, query, result)

    def remove_bookmark(self, name: str) -> bool:
        """Remove a bookmark."""
        return self.history.remove_bookmark(name)

    def get_frequent_patterns(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most frequently searched patterns."""
        return self.history.get_frequent_patterns(limit)

    def get_recent_patterns(self, days: int = 7, limit: int = 20) -> list[str]:
        """Get recently used search patterns."""
        return self.history.get_recent_patterns(days, limit)

    # Enhanced history and session management
    def get_current_session(self) -> Any | None:
        """Get the current search session."""
        return self.history.get_current_session()

    def get_search_sessions(self, limit: int | None = None) -> list[Any]:
        """Get search sessions, most recent first."""
        return self.history.get_sessions(limit)  # type: ignore[no-any-return]

    def end_current_session(self) -> None:
        """Manually end the current search session."""
        self.history.end_current_session()

    def get_search_analytics(self, days: int = 30) -> dict[str, Any]:
        """Get comprehensive search analytics for the specified period."""
        return self.history.get_search_analytics(days)

    def get_pattern_suggestions(self, partial_pattern: str, limit: int = 5) -> list[str]:
        """Get search pattern suggestions based on history."""
        return self.history.get_pattern_suggestions(partial_pattern, limit)

    def rate_last_search(self, pattern: str, rating: int) -> bool:
        """Rate a search result (1-5 stars)."""
        return self.history.rate_search(pattern, rating)

    def add_search_tags(self, pattern: str, tags: list[str]) -> bool:
        """Add tags to a search in history."""
        return self.history.add_tags_to_search(pattern, set(tags))

    def search_history_by_tags(self, tags: list[str]) -> list[Any]:
        """Find searches by tags."""
        return self.history.search_history_by_tags(set(tags))

    # Enhanced bookmark organization
    def create_bookmark_folder(self, name: str, description: str | None = None) -> bool:
        """Create a new bookmark folder."""
        return self.history.create_folder(name, description)

    def delete_bookmark_folder(self, name: str) -> bool:
        """Delete a bookmark folder."""
        return self.history.delete_folder(name)

    def add_bookmark_to_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Add a bookmark to a folder."""
        return self.history.add_bookmark_to_folder(bookmark_name, folder_name)

    def remove_bookmark_from_folder(self, bookmark_name: str, folder_name: str) -> bool:
        """Remove a bookmark from a folder."""
        return self.history.remove_bookmark_from_folder(bookmark_name, folder_name)

    def get_bookmark_folders(self) -> dict[str, Any]:
        """Get all bookmark folders."""
        return self.history.get_folders()

    def get_bookmarks_in_folder(self, folder_name: str) -> list[Any]:
        """Get all bookmarks in a specific folder."""
        return self.history.get_bookmarks_in_folder(folder_name)

    def get_indexer_stats(self) -> dict[str, Any]:
        """Get indexer cache statistics."""
        return self.indexer.get_cache_stats()

    # Error reporting and diagnostics
    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of errors encountered during search operations."""
        return self.error_collector.get_summary()

    def get_error_report(self) -> str:
        """Get detailed error report."""
        return create_error_report(self.error_collector)

    def get_errors_by_category(self, category: str) -> list[Any]:
        """Get errors of a specific category."""
        from ..utils.error_handling import ErrorCategory

        try:
            error_category = ErrorCategory(category)
            return self.error_collector.get_errors_by_category(error_category)
        except ValueError:
            return []

    def clear_errors(self) -> None:
        """Clear all collected errors."""
        self.error_collector.clear()

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return self.error_collector.has_critical_errors()

    def suppress_error_category(self, category: str) -> None:
        """Suppress errors of a specific category."""
        from ..utils.error_handling import ErrorCategory

        try:
            error_category = ErrorCategory(category)
            self.error_collector.suppress_category(error_category)
        except ValueError:
            pass

    def configure_logging(
        self,
        level: str = "INFO",
        format_type: str = "simple",
        log_file: str | None = None,
        enable_file: bool = False,
    ) -> None:
        """Configure logging settings."""
        from ..utils.logging_config import LogFormat, LogLevel, configure_logging

        try:
            log_level = LogLevel(level.upper())
            log_format = LogFormat(format_type.lower())
            log_path = Path(log_file) if log_file else None

            self.logger = configure_logging(
                level=log_level, format_type=log_format, log_file=log_path, enable_file=enable_file
            )
        except ValueError as e:
            self.logger.warning(f"Invalid logging configuration: {e}")

    def enable_debug_logging(self) -> None:
        """Enable debug logging for troubleshooting."""
        from ..utils.logging_config import enable_debug_logging

        enable_debug_logging()
        self.logger.info("Debug logging enabled")

    def disable_logging(self) -> None:
        """Disable all logging."""
        from ..utils.logging_config import disable_logging

        disable_logging()

    # Advanced ranking and result organization
    def search_with_ranking(
        self,
        pattern: str,
        ranking_strategy: str = "hybrid",
        cluster_results: bool = False,
        **kwargs: Any,
    ) -> SearchResult:
        """
        Search with configurable ranking strategy.

        Args:
            pattern: Search pattern
            ranking_strategy: 'relevance', 'frequency', 'recency', 'popularity', or 'hybrid'
            cluster_results: Whether to cluster similar results
            **kwargs: Additional search parameters
        """
        # Parse ranking strategy
        try:
            strategy = RankingStrategy(ranking_strategy.lower())
        except ValueError:
            strategy = RankingStrategy.HYBRID

        # Perform basic search
        result = self.search(pattern, **kwargs)

        # Re-sort with specified strategy
        try:
            sorted_items = sort_items(result.items, self.cfg, pattern, strategy)

            # Cluster results if requested
            if cluster_results:
                clusters = cluster_results_by_similarity(sorted_items)
                # Flatten clusters but maintain cluster order
                clustered_items = []
                for cluster in clusters:
                    clustered_items.extend(cluster)
                sorted_items = clustered_items

            # Create new result with sorted items
            result.items = sorted_items

        except Exception as e:
            self.logger.error(f"Error in advanced ranking: {e}")
            self.error_collector.add_error(e)

        return result

    def get_result_clusters(
        self, result: SearchResult, similarity_threshold: float = 0.8
    ) -> list[list[SearchItem]]:
        """
        Cluster search results by content similarity.

        Args:
            result: Search result to cluster
            similarity_threshold: Minimum similarity for clustering (0.0-1.0)

        Returns:
            List of clusters, each containing similar items
        """
        try:
            return cluster_results_by_similarity(result.items, similarity_threshold)
        except Exception as e:
            self.logger.error(f"Error clustering results: {e}")
            return [result.items] if result.items else []

    def get_ranking_suggestions(self, pattern: str, results: SearchResult) -> dict[str, Any]:
        """
        Get suggestions for optimal ranking strategy based on query and results.

        Args:
            pattern: Search pattern
            results: Search results to analyze

        Returns:
            Dictionary with ranking suggestions and analysis
        """
        analysis: dict[str, Any] = {
            "query_type": "unknown",
            "recommended_strategy": "hybrid",
            "result_diversity": 0.0,
            "file_spread": 0,
            "suggestions": [],
        }

        try:
            # Analyze query characteristics
            if len(pattern.split()) > 3:
                analysis["query_type"] = "complex"
                analysis["recommended_strategy"] = "relevance"
                analysis["suggestions"].append(
                    "Complex queries benefit from relevance-based ranking"
                )
            elif pattern.isupper():
                analysis["query_type"] = "constant"
                analysis["recommended_strategy"] = "frequency"
                analysis["suggestions"].append(
                    "Constant names often benefit from frequency-based ranking"
                )
            elif any(char in pattern for char in ["_", "-", "."]):
                analysis["query_type"] = "identifier"
                analysis["recommended_strategy"] = "hybrid"
                analysis["suggestions"].append("Identifiers work well with hybrid ranking")
            else:
                analysis["query_type"] = "simple"
                analysis["recommended_strategy"] = "popularity"
                analysis["suggestions"].append(
                    "Simple queries can benefit from popularity-based ranking"
                )

            # Analyze result characteristics
            if results.items:
                unique_files = len({item.file for item in results.items})
                analysis["file_spread"] = unique_files

                if unique_files > 10:
                    analysis["suggestions"].append("Many files found - consider clustering results")

                # Calculate result diversity
                if len(results.items) > 1:
                    clusters = cluster_results_by_similarity(results.items)
                    analysis["result_diversity"] = len(clusters) / len(results.items)

                    if analysis["result_diversity"] < 0.3:
                        analysis["suggestions"].append("Low diversity - results are very similar")
                    elif analysis["result_diversity"] > 0.8:
                        analysis["suggestions"].append(
                            "High diversity - results are quite different"
                        )

        except Exception as e:
            self.logger.error(f"Error analyzing ranking suggestions: {e}")
            analysis["suggestions"].append("Error analyzing results")

        return analysis

    def cleanup_old_cache_entries(self, days_old: int = 30) -> int:
        """Clean up old cache entries."""
        return self.indexer.cleanup_old_entries(days_old)
