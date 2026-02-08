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

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..indexing.indexer import Indexer
from ..integrations.multi_repo import MultiRepoSearchResult, RepositoryInfo
from ..search.boolean import evaluate_boolean_query_with_items, extract_terms, parse_boolean_query
from ..search.matchers import search_in_file
from ..search.scorer import (
    RankingStrategy,
    cluster_results_by_similarity,
    deduplicate_overlapping_results,
    sort_items,
)
from ..storage.qdrant_client import QdrantConfig
from ..storage.vector_db import EmbeddingConfig, MultiProviderVectorManager
from ..indexing.cache import CacheManager
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
from .managers.distributed_indexing_integration import DistributedIndexingManager
from .managers.ide_integration import IDEIntegrationManager
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
        self.hybrid_search_manager = HybridSearchManager(self.cfg)
        self.cache_integration = CacheIntegrationManager(self.cfg)
        self.dependency_integration = DependencyIntegrationManager(self.cfg)
        self.file_watching = FileWatchingManager(self.cfg)
        self.graphrag_integration = GraphRAGIntegrationManager(self.cfg, qdrant_config)
        self.indexing_integration = IndexingIntegrationManager(self.cfg)
        self.distributed_indexing = DistributedIndexingManager(self.cfg)
        self.ide_integration = IDEIntegrationManager(self.cfg)
        self.multi_repo_integration = MultiRepoIntegrationManager(self.cfg)
        self.parallel_processing = ParallelSearchManager(self.cfg)

        # Set up dependencies between managers
        self.hybrid_search_manager.set_dependencies(self.error_collector, self.logger)
        self.graphrag_integration.set_dependencies(self.logger, self.error_collector)
        self.indexing_integration.set_dependencies(self.logger, self.error_collector)
        self.dependency_integration.set_logger(self.logger)
        self.file_watching.set_indexer(self.indexer)
        self.file_watching.set_cache_invalidation_callback(self._on_files_changed)
        self.parallel_processing.set_logger(self.logger)

    def _on_files_changed(self, changed_paths: list[Path]) -> None:
        """Callback invoked by the file watcher when files change.

        Invalidates file content and search result caches so that subsequent
        searches return up-to-date results.
        """
        for file_path in changed_paths:
            self.cache_integration.invalidate_file_cache(file_path)
        # Also invalidate the legacy search result cache entirely since we
        # cannot cheaply determine which cached results depend on which files
        # in the legacy (non-CacheManager) path.
        self.cache_integration._search_result_cache.clear()

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
                # Extract individual terms and search for each to build candidate items
                terms = extract_terms(boolean_query)
                all_items: list[SearchItem] = []
                seen_spans: set[tuple[int, int]] = set()
                for term in terms:
                    # Use case-insensitive regex to match boolean evaluator behaviour
                    escaped_term = f"(?i){re.escape(term)}"
                    term_query = Query(
                        pattern=escaped_term,
                        use_regex=True,
                        use_boolean=False,
                        context=query.context,
                        output=query.output,
                        filters=query.filters,
                        metadata_filters=query.metadata_filters,
                        search_docstrings=query.search_docstrings,
                        search_comments=query.search_comments,
                        search_strings=query.search_strings,
                        count_only=query.count_only,
                        max_per_file=None,
                    )
                    term_items = search_in_file(path, text, term_query)
                    for item in term_items:
                        key = (item.start_line, item.end_line)
                        if key not in seen_spans:
                            seen_spans.add(key)
                            all_items.append(item)
                # Apply boolean logic to filter the collected items
                items = evaluate_boolean_query_with_items(boolean_query, text, all_items)
            except Exception as e:
                self.logger.warning(f"Boolean query error in {path}: {e}")
                return []
        else:
            items = search_in_file(path, text, query)

        # Apply per-file limit if specified
        if query.max_per_file is not None and len(items) > query.max_per_file:
            items = items[: query.max_per_file]

        return items

    def _get_cached_file_content(self, path: Path) -> str | None:
        """
        Get file content with caching based on modification time.

        This method delegates to the cache integration manager for file content caching.

        Args:
            path: Path to the file to read

        Returns:
            File content as string if successful, None if file cannot be read
        """
        return self.cache_integration.get_cached_file_content(path)

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

        # When auto-watch is active the index is kept up-to-date
        # incrementally, so we can skip the expensive full scan.
        if self.file_watching.is_auto_watch_enabled():
            self.indexer.load()  # ensure index is loaded
            total_seen = self.indexer.count_indexed()
        else:
            try:
                changed, removed, total_seen = self.indexer.scan()
                self.indexer.save()
                self.logger.debug(
                    f"Indexer scan: {total_seen} files seen, {len(changed or [])} changed"
                )
            except Exception as e:
                self.logger.error(f"Error during indexing: {e}")
                self.error_collector.add_error(e)
                total_seen = 0

        paths = list(self.indexer.iter_all_paths())
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
            max_per_file=kwargs.get("max_per_file"),
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
        return await self.hybrid_search_manager.hybrid_search(
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
        changed = []
        removed = []
        total_seen = 0
        try:
            changed, removed, total_seen = self.indexer.scan()
            self.logger.debug(
                f"Indexer scan: {len(changed)} changed, {len(removed)} removed, {total_seen} total"
            )
        except Exception as e:
            self.logger.error(f"Error during indexing for semantic search: {e}")
            self.error_collector.add_error(e)

        paths = list(self.indexer.iter_all_paths())

        # Delegate to hybrid search manager
        return self.hybrid_search_manager.search_semantic(
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

    def check_dependency_path(
        self, source: str, target: str, graph: Any | None = None
    ) -> bool:
        """
        Check if there is a dependency path from source module to target module.

        This uses graph traversal (BFS) to determine if a transitive dependency
        exists between two modules in the dependency graph.

        Args:
            source: Source module name
            target: Target module name
            graph: Dependency graph to use (if None, analyzes current project)

        Returns:
            True if a dependency path exists from source to target

        Example:
            >>> has_path = engine.check_dependency_path("src.core.api", "src.utils.helpers")
            >>> print(f"Dependency path exists: {has_path}")
        """
        return self.dependency_integration.check_dependency_path(source, target, graph)

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
        result = self.cache_integration.enable_caching(
            backend=backend,
            cache_dir=cache_dir,
            max_size=max_size,
            default_ttl=default_ttl,
            compression=compression,
        )
        if result:
            self.logger.info(f"Caching enabled with {backend} backend")
        else:
            self.logger.error("Failed to enable caching")
        return result

    def disable_caching(self) -> None:
        """
        Disable search result caching.

        Stops caching new results and optionally clears existing cache.

        Example:
            >>> engine.disable_caching()
            >>> print("Caching disabled")
        """
        self.cache_integration.disable_caching()
        self.logger.info("Caching disabled")

    def is_caching_enabled(self) -> bool:
        """
        Check if search result caching is enabled.

        Returns:
            True if caching is enabled, False otherwise
        """
        return self.cache_integration.is_caching_enabled()

    def clear_cache(self) -> None:
        """
        Clear all cached search results.

        Example:
            >>> engine.clear_cache()
            >>> print("Cache cleared")
        """
        self.cache_integration.clear_caches()
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
        return self.cache_integration.get_cache_stats()

    def invalidate_cache_for_file(self, file_path: Path | str) -> None:
        """
        Invalidate cached results that depend on a specific file.

        This is automatically called when file watching detects changes,
        but can be manually triggered when needed.

        Args:
            file_path: Path of the file that changed

        Example:
            >>> # Manually invalidate cache for a changed file
            >>> engine.invalidate_cache_for_file("src/main.py")
        """
        self.cache_integration.invalidate_file_cache(Path(file_path))

    def set_cache_ttl(self, ttl: float) -> None:
        """
        Set cache time-to-live in seconds.

        Args:
            ttl: Time-to-live in seconds for cache entries
        """
        self.cache_integration.set_cache_ttl(ttl)

    def get_cache_hit_rate(self) -> float:
        """
        Get cache hit rate as a percentage.

        Returns:
            Cache hit rate percentage (0.0-100.0)
        """
        return self.cache_integration.get_cache_hit_rate()

    def _generate_cache_key(self, query: Query) -> str:
        """Generate a cache key for a search query."""
        return self.cache_integration._generate_cache_key(query)

    def _get_file_dependencies(self, result: SearchResult) -> set[str]:
        """Extract file dependencies from search result."""
        return {str(p) for p in self.cache_integration._get_file_dependencies(result)}

    # ── Distributed Indexing ──

    def enable_distributed_indexing(
        self, num_workers: int | None = None, max_queue_size: int = 10000
    ) -> bool:
        """
        Enable distributed indexing for large codebases.

        Args:
            num_workers: Number of worker processes (defaults to min(cpu_count, 8))
            max_queue_size: Maximum work queue size

        Returns:
            True if distributed indexing was enabled successfully, False otherwise

        Example:
            >>> engine.enable_distributed_indexing(num_workers=4)
            >>> import asyncio
            >>> asyncio.run(engine.distributed_index_codebase(["/path/to/project"]))
        """
        result = self.distributed_indexing.enable_distributed_indexing(
            num_workers=num_workers, max_queue_size=max_queue_size
        )
        if result:
            self.logger.info("Distributed indexing enabled")
        else:
            self.logger.error("Failed to enable distributed indexing")
        return result

    def disable_distributed_indexing(self) -> None:
        """Disable distributed indexing and stop all workers."""
        self.distributed_indexing.disable_distributed_indexing()
        self.logger.info("Distributed indexing disabled")

    def is_distributed_indexing_enabled(self) -> bool:
        """Check if distributed indexing is enabled."""
        return self.distributed_indexing.is_distributed_enabled()

    async def distributed_index_codebase(
        self,
        directories: list[str],
        branch: str | None = None,
        repo_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Index codebase using distributed workers.

        Args:
            directories: Directories to index
            branch: Git branch name
            repo_name: Repository name

        Returns:
            List of progress update dictionaries

        Example:
            >>> import asyncio
            >>> updates = asyncio.run(engine.distributed_index_codebase(["."]))
            >>> for u in updates:
            ...     print(f"{u['progress']:.0%} - {u['description']}")
        """
        if not self.distributed_indexing.is_distributed_enabled():
            self.logger.error("Distributed indexing not enabled")
            return []
        return await self.distributed_indexing.index_codebase(
            directories=directories, branch=branch, repo_name=repo_name
        )

    async def get_distributed_worker_stats(self) -> list[dict[str, Any]]:
        """
        Get statistics for all distributed indexing workers.

        Returns:
            List of worker statistics dictionaries
        """
        if not self.distributed_indexing.is_distributed_enabled():
            return []
        return await self.distributed_indexing.get_worker_stats()

    async def scale_distributed_workers(self, target_count: int) -> bool:
        """
        Dynamically scale the number of distributed indexing workers.

        Args:
            target_count: Target number of workers

        Returns:
            True if scaling succeeded, False otherwise
        """
        if not self.distributed_indexing.is_distributed_enabled():
            self.logger.error("Distributed indexing not enabled")
            return False
        return await self.distributed_indexing.scale_workers(target_count)

    async def get_distributed_performance_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive performance metrics for distributed indexing.

        Returns:
            Dictionary with worker, queue, and performance metrics
        """
        if not self.distributed_indexing.is_distributed_enabled():
            return {}
        return await self.distributed_indexing.get_performance_metrics()

    def get_distributed_queue_stats(self) -> dict[str, Any]:
        """
        Get work queue statistics (synchronous).

        Returns:
            Dictionary with queue statistics
        """
        if not self.distributed_indexing.is_distributed_enabled():
            return {}
        return self.distributed_indexing.get_queue_stats()

    # ── IDE Integration ──

    def enable_ide_integration(self) -> bool:
        """
        Enable IDE integration features (jump-to-definition, find-references, etc.).

        Returns:
            True if IDE integration was enabled successfully, False otherwise

        Example:
            >>> engine.enable_ide_integration()
            >>> loc = engine.jump_to_definition("src/main.py", 10, "my_func")
        """
        result = self.ide_integration.enable_ide_integration(self)
        if result:
            self.logger.info("IDE integration enabled")
        else:
            self.logger.error("Failed to enable IDE integration")
        return result

    def disable_ide_integration(self) -> None:
        """Disable IDE integration features."""
        self.ide_integration.disable_ide_integration()
        self.logger.info("IDE integration disabled")

    def is_ide_enabled(self) -> bool:
        """Check if IDE integration is enabled."""
        return self.ide_integration.is_ide_enabled()

    def jump_to_definition(
        self, file_path: str, line: int, symbol: str
    ) -> dict[str, Any] | None:
        """
        Find the definition of a symbol.

        Args:
            file_path: File requesting the jump
            line: Line number where the symbol appears
            symbol: The identifier to look up

        Returns:
            Dictionary with definition location, or None if not found

        Example:
            >>> loc = engine.jump_to_definition("src/main.py", 10, "my_func")
            >>> if loc:
            ...     print(f"Definition at {loc['file']}:{loc['line']}")
        """
        if not self.ide_integration.is_ide_enabled():
            self.logger.error("IDE integration not enabled")
            return None
        return self.ide_integration.jump_to_definition(file_path, line, symbol)

    def find_references(
        self,
        file_path: str,
        line: int,
        symbol: str,
        include_definition: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Find all references to a symbol across the codebase.

        Args:
            file_path: Originating file
            line: Originating line
            symbol: The identifier to search for
            include_definition: Whether to include the definition itself

        Returns:
            List of reference location dictionaries
        """
        if not self.ide_integration.is_ide_enabled():
            self.logger.error("IDE integration not enabled")
            return []
        return self.ide_integration.find_references(
            file_path, line, symbol, include_definition
        )

    def provide_completion(
        self, file_path: str, line: int, column: int, prefix: str = ""
    ) -> list[dict[str, Any]]:
        """
        Provide auto-completion suggestions for the given cursor position.

        Args:
            file_path: Current file
            line: Cursor line
            column: Cursor column
            prefix: Partially typed identifier

        Returns:
            List of completion item dictionaries
        """
        if not self.ide_integration.is_ide_enabled():
            return []
        return self.ide_integration.provide_completion(file_path, line, column, prefix)

    def provide_hover(
        self, file_path: str, line: int, column: int, symbol: str
    ) -> dict[str, Any] | None:
        """
        Provide hover information for a symbol.

        Args:
            file_path: Current file
            line: Cursor line
            column: Cursor column
            symbol: The hovered identifier

        Returns:
            Dictionary with hover information, or None
        """
        if not self.ide_integration.is_ide_enabled():
            return None
        return self.ide_integration.provide_hover(file_path, line, column, symbol)

    def get_document_symbols(self, file_path: str) -> list[dict[str, Any]]:
        """
        List all symbols (functions, classes, variables) in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of document symbol dictionaries
        """
        if not self.ide_integration.is_ide_enabled():
            return []
        return self.ide_integration.get_document_symbols(file_path)

    def get_workspace_symbols(self, query: str = "") -> list[dict[str, Any]]:
        """
        Search for symbols across the entire workspace.

        Args:
            query: Optional filter string for symbol names

        Returns:
            List of document symbol dictionaries
        """
        if not self.ide_integration.is_ide_enabled():
            return []
        return self.ide_integration.get_workspace_symbols(query)

    def get_diagnostics(self, file_path: str) -> list[dict[str, Any]]:
        """
        Run lightweight diagnostics on a file.

        Currently checks for TODO/FIXME/HACK markers and self-imports.

        Args:
            file_path: The file to diagnose

        Returns:
            List of diagnostic dictionaries
        """
        if not self.ide_integration.is_ide_enabled():
            return []
        return self.ide_integration.get_diagnostics(file_path)

    def ide_structured_query(self, query: Query) -> dict[str, Any]:
        """
        Structured query interface for IDE consumption.

        Returns a JSON-serialisable dict with search results and stats.

        Args:
            query: A Query object for searching

        Returns:
            Dictionary with 'items' and 'stats' keys
        """
        if not self.ide_integration.is_ide_enabled():
            return {"items": [], "stats": {}}
        return self.ide_integration.ide_query(query)

    # ── Multi-Repository Search ──

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
        result = self.multi_repo_integration.enable_multi_repo(max_workers=max_workers)
        if result:
            self.logger.info("Multi-repository search enabled")
        else:
            self.logger.error("Failed to enable multi-repository search")
        return result

    def disable_multi_repo(self) -> None:
        """
        Disable multi-repository search capabilities.

        Example:
            >>> engine.disable_multi_repo()
            >>> print("Multi-repository search disabled")
        """
        self.multi_repo_integration.disable_multi_repo()
        self.logger.info("Multi-repository search disabled")

    def is_multi_repo_enabled(self) -> bool:
        """
        Check if multi-repository search is enabled.

        Returns:
            True if multi-repo is enabled, False otherwise
        """
        return self.multi_repo_integration.is_multi_repo_enabled()

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
        if not self.multi_repo_integration.is_multi_repo_enabled():
            self.logger.error("Multi-repository search not enabled")
            return False

        return self.multi_repo_integration.add_repository(
            name=name, path=str(path), config=config, priority=priority, **metadata
        )

    def configure_repository(self, name: str, **config_updates: Any) -> bool:
        """
        Update repository configuration in multi-repository search.

        Args:
            name: Repository name
            **config_updates: Configuration updates (priority, enabled, etc.)

        Returns:
            True if configuration was updated, False otherwise

        Example:
            >>> engine.configure_repository(
            ...     "web-app",
            ...     priority="high",
            ...     enabled=True,
            ... )
        """
        if not self.multi_repo_integration.is_multi_repo_enabled():
            self.logger.error("Multi-repository search not enabled")
            return False

        return self.multi_repo_integration.configure_repository(name, **config_updates)

    def remove_repository(self, name: str) -> bool:
        """
        Remove a repository from multi-repository search.

        Args:
            name: Name of the repository to remove

        Returns:
            True if repository was removed, False if not found
        """
        return self.multi_repo_integration.remove_repository(name)

    def list_repositories(self) -> list[str]:
        """
        Get list of repository names in multi-repository search.

        Returns:
            List of repository names, empty if multi-repo not enabled
        """
        return self.multi_repo_integration.list_repositories()

    def get_repository_info(self, name: str) -> RepositoryInfo | None:
        """
        Get detailed information about a repository.

        Args:
            name: Repository name

        Returns:
            RepositoryInfo if found, None otherwise
        """
        if not self.multi_repo_integration.is_multi_repo_enabled():
            return None

        engine = self.multi_repo_integration.multi_repo_engine
        if engine:
            return engine.get_repository_info(name)  # type: ignore[no-any-return]
        return None

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
        if not self.multi_repo_integration.is_multi_repo_enabled():
            self.logger.error("Multi-repository search not enabled")
            return None

        engine = self.multi_repo_integration.multi_repo_engine
        if not engine:
            return None

        return engine.search_all(  # type: ignore[no-any-return]
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
        result = self.multi_repo_integration.search_repositories(
            query=query, repositories=repositories
        )
        return result

    def get_multi_repo_health(self) -> dict[str, Any]:
        """
        Get health status for all repositories in multi-repository system.

        Returns:
            Dictionary with health information, empty if not enabled
        """
        return self.multi_repo_integration.get_repository_health()

    def get_multi_repo_stats(self) -> dict[str, Any]:
        """
        Get search performance statistics for multi-repository system.

        Returns:
            Dictionary with search statistics, empty if not enabled
        """
        return self.multi_repo_integration.get_repository_stats()

    def sync_repositories(self) -> dict[str, bool]:
        """
        Synchronize all repositories (refresh status and health).

        Returns:
            Dictionary mapping repository names to sync success status
        """
        return self.multi_repo_integration.sync_repositories()

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

    def word_level_fuzzy_search(
        self,
        pattern: str,
        max_distance: int = 2,
        min_similarity: float = 0.6,
        algorithms: list[str] | None = None,
        max_results: int = 1000,
        **kwargs: Any,
    ) -> SearchResult:
        """
        Word-level fuzzy search using actual similarity algorithms.

        Unlike regex-based fuzzy search, this method compares individual words
        in file content against the pattern using real edit-distance and
        similarity algorithms, returning matches with precise similarity scores.

        Args:
            pattern: Word or short phrase to search for
            max_distance: Maximum edit distance for distance-based algorithms
            min_similarity: Minimum similarity score (0.0 to 1.0)
            algorithms: List of algorithm names to use (default: all)
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
        """
        from ..search.fuzzy import FuzzyAlgorithm, FuzzyMatch, fuzzy_search_advanced

        # Parse algorithms
        fuzzy_algos: list[FuzzyAlgorithm] | None = None
        if algorithms:
            fuzzy_algos = []
            for algo_name in algorithms:
                try:
                    fuzzy_algos.append(FuzzyAlgorithm(algo_name.lower()))
                except ValueError:
                    continue

        t0 = time.perf_counter()

        # Ensure index is up-to-date
        if self.file_watching.is_auto_watch_enabled():
            self.indexer.load()
        else:
            try:
                self.indexer.scan()
            except Exception as e:
                self.error_collector.add_error(e)

        paths = list(self.indexer.iter_all_paths())
        all_items: list[SearchItem] = []
        seen_keys: set[tuple[str, int]] = set()
        files_matched_set: set[Path] = set()

        for path in paths:
            text = self._get_cached_file_content(path)
            if text is None:
                continue

            matches: list[FuzzyMatch] = fuzzy_search_advanced(
                text=text,
                pattern=pattern,
                algorithms=fuzzy_algos,
                max_distance=max_distance,
                min_similarity=min_similarity,
                combine_results=True,
            )

            if not matches:
                continue

            files_matched_set.add(path)
            lines = text.splitlines()

            for match in matches:
                # Find line number for this match
                line_num = text[:match.start].count("\n") + 1
                key = (str(path), line_num)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Build context window
                from ..utils.helpers import extract_context, split_lines_keepends

                ctx_lines = split_lines_keepends(text)
                ctx_s, ctx_e, slice_lines = extract_context(
                    ctx_lines, line_num, line_num,
                    window=kwargs.get("context", self.cfg.context),
                )

                # Calculate column within line
                line_start_offset = sum(len(l) for l in lines[:line_num - 1]) + (line_num - 1)
                col_start = match.start - line_start_offset
                col_end = match.end - line_start_offset

                item = SearchItem(
                    file=path,
                    start_line=ctx_s,
                    end_line=ctx_e,
                    lines=slice_lines,
                    match_spans=[(line_num - ctx_s, (max(0, col_start), max(0, col_end)))],
                )
                all_items.append(item)

            if len(all_items) >= max_results:
                break

        # Sort and deduplicate
        all_items = sort_items(all_items, self.cfg, pattern)
        all_items = deduplicate_overlapping_results(all_items)
        all_items = all_items[:max_results]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return SearchResult(
            items=all_items,
            stats=SearchStats(
                files_scanned=len(paths),
                files_matched=len(files_matched_set),
                items=len(all_items),
                elapsed_ms=elapsed_ms,
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

    def suggest_corrections(
        self,
        word: str,
        max_suggestions: int = 5,
        algorithm: str = "damerau_levenshtein",
    ) -> list[tuple[str, float]]:
        """
        Suggest spelling corrections for a word based on identifiers found in the codebase.

        Scans all indexed files, extracts unique identifiers, and returns the
        most similar ones to the given word using the specified fuzzy algorithm.

        Args:
            word: Word to find corrections for
            max_suggestions: Maximum number of suggestions to return
            algorithm: Fuzzy algorithm to use for similarity calculation

        Returns:
            List of (suggestion, similarity_score) tuples sorted by similarity

        Example:
            >>> suggestions = engine.suggest_corrections("conection")
            >>> for suggestion, score in suggestions:
            ...     print(f"  {suggestion} (similarity: {score:.2f})")
        """
        from ..search.fuzzy import FuzzyAlgorithm, suggest_corrections as _suggest

        # Parse algorithm
        try:
            fuzzy_algo = FuzzyAlgorithm(algorithm.lower())
        except ValueError:
            fuzzy_algo = FuzzyAlgorithm.DAMERAU_LEVENSHTEIN

        # Build dictionary from codebase identifiers
        import re as _re

        identifier_set: set[str] = set()

        # Ensure index is up-to-date
        if self.file_watching.is_auto_watch_enabled():
            self.indexer.load()
        else:
            try:
                self.indexer.scan()
            except Exception as e:
                self.error_collector.add_error(e)

        paths = list(self.indexer.iter_all_paths())
        for path in paths[:500]:  # Limit to avoid excessive processing
            text = self._get_cached_file_content(path)
            if text is None:
                continue
            # Extract identifiers (words with at least 3 chars)
            identifiers = _re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text)
            identifier_set.update(identifiers)

        dictionary = list(identifier_set)
        return _suggest(
            word=word,
            dictionary=dictionary,
            max_suggestions=max_suggestions,
            algorithm=fuzzy_algo,
        )

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

    def check_boolean_match(self, query_str: str, file_path: Path | str) -> bool:
        """
        Check if a file matches a boolean query expression.

        This provides a quick boolean check without returning detailed match
        locations — useful for filtering files or pre-screening content.

        Args:
            query_str: Boolean query string (e.g., "(async AND handler) NOT test")
            file_path: Path to the file to check

        Returns:
            True if the file content matches the boolean expression

        Example:
            >>> if engine.check_boolean_match("(database AND connection) NOT test", "db.py"):
            ...     print("File matches!")
        """
        from ..search.boolean import evaluate_boolean_query, parse_boolean_query

        path = Path(file_path)
        content = self._get_cached_file_content(path)
        if content is None:
            return False

        try:
            boolean_query = parse_boolean_query(query_str)
            return evaluate_boolean_query(boolean_query, content)
        except Exception as e:
            self.logger.warning(f"Boolean query error: {e}")
            return False

    def get_results_grouped_by_file(
        self, result: SearchResult
    ) -> dict[Path, list[SearchItem]]:
        """
        Group search results by file for organized output.

        Each file's results are sorted by line number, making it easy to
        process results on a per-file basis.

        Args:
            result: Search result to group

        Returns:
            Dictionary mapping file paths to their sorted search items

        Example:
            >>> result = engine.search("def ")
            >>> grouped = engine.get_results_grouped_by_file(result)
            >>> for file_path, items in grouped.items():
            ...     print(f"{file_path}: {len(items)} matches")
        """
        from ..search.scorer import group_results_by_file

        return group_results_by_file(result.items)

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

    # ── Dependency integration: expose all manager methods ──

    def detect_circular_dependencies(self, graph: Any | None = None) -> list[list[str]]:
        """
        Detect circular dependencies in the codebase.

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            List of circular dependency chains
        """
        return self.dependency_integration.detect_circular_dependencies(graph)

    def get_module_coupling_metrics(self, graph: Any | None = None) -> dict[str, Any]:
        """
        Calculate module coupling metrics (afferent/efferent coupling, instability).

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            Dictionary with coupling metrics for each module
        """
        return self.dependency_integration.get_module_coupling_metrics(graph)

    def find_dead_code(self, graph: Any | None = None) -> list[str]:
        """
        Identify potentially dead code (unused modules).

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            List of module names that appear to be unused
        """
        return self.dependency_integration.find_dead_code(graph)

    def export_dependency_graph(self, graph: Any, format: str = "dot") -> str:
        """
        Export dependency graph in specified format.

        Args:
            graph: Dependency graph to export
            format: Export format ("dot", "json", "csv")

        Returns:
            String representation of the graph in specified format
        """
        return self.dependency_integration.export_dependency_graph(graph, format)

    # ── File watching: expose all manager methods ──

    def pause_watching(self) -> None:
        """Temporarily pause all file watchers."""
        self.file_watching.pause_watching()

    def resume_watching(self) -> None:
        """Resume all paused file watchers."""
        self.file_watching.resume_watching()

    def get_watcher_status(self, name: str) -> dict[str, Any]:
        """
        Get status information for a specific watcher.

        Args:
            name: Name of the watcher

        Returns:
            Dictionary with watcher status information
        """
        return self.file_watching.get_watcher_status(name)

    def set_watch_filters(
        self, include_patterns: list[str] | None = None, exclude_patterns: list[str] | None = None
    ) -> None:
        """
        Set file patterns to include or exclude from watching.

        Args:
            include_patterns: Patterns for files to include in watching
            exclude_patterns: Patterns for files to exclude from watching
        """
        self.file_watching.set_watch_filters(include_patterns, exclude_patterns)

    def force_rescan(self) -> bool:
        """
        Force a rescan of all watched directories.

        Returns:
            True if rescan was successful, False otherwise
        """
        return self.file_watching.force_rescan()

    def get_watch_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for file watching operations.

        Returns:
            Dictionary with performance metrics
        """
        return self.file_watching.get_watch_performance_metrics()

    # ── GraphRAG integration: expose all manager methods ──

    def is_graphrag_initialized(self) -> bool:
        """Check if GraphRAG is initialized."""
        return self.graphrag_integration.is_initialized()

    def is_graphrag_enabled(self) -> bool:
        """Check if GraphRAG is enabled."""
        return self.graphrag_integration.is_enabled()

    def get_graph_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        return self.graphrag_integration.get_graph_stats()

    def get_vector_store_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        return self.graphrag_integration.get_vector_store_stats()

    async def add_graph_entities(self, entities: list[Any]) -> bool:
        """Add entities to the knowledge graph."""
        return await self.graphrag_integration.add_entities(entities)

    async def add_graph_relationships(self, relationships: list[Any]) -> bool:
        """Add relationships to the knowledge graph."""
        return await self.graphrag_integration.add_relationships(relationships)

    async def find_similar_entities(self, entity_id: str, limit: int = 10) -> list[Any]:
        """Find entities similar to the given entity."""
        return await self.graphrag_integration.find_similar_entities(entity_id, limit)

    async def get_entity_context(self, entity_id: str, max_hops: int = 2) -> dict[str, Any]:
        """Get contextual information for an entity."""
        return await self.graphrag_integration.get_entity_context(entity_id, max_hops)

    async def update_graph_entity(self, entity_id: str, updates: dict[str, Any]) -> bool:
        """Update an entity in the knowledge graph."""
        return await self.graphrag_integration.update_entity(entity_id, updates)

    async def delete_graph_entity(self, entity_id: str) -> bool:
        """Delete an entity from the knowledge graph."""
        return await self.graphrag_integration.delete_entity(entity_id)

    async def export_knowledge_graph(self, format: str = "json") -> str:
        """Export the knowledge graph in specified format."""
        return await self.graphrag_integration.export_graph(format)

    async def import_knowledge_graph(self, data: str, format: str = "json") -> bool:
        """Import a knowledge graph from data."""
        return await self.graphrag_integration.import_graph(data, format)

    # ── Indexing integration: expose all manager methods ──

    def is_metadata_index_initialized(self) -> bool:
        """Check if metadata indexing is initialized."""
        return self.indexing_integration.is_initialized()

    def is_metadata_index_enabled(self) -> bool:
        """Check if metadata indexing is enabled."""
        return self.indexing_integration.is_enabled()

    def get_metadata_index_stats(self) -> dict[str, Any]:
        """Get metadata index statistics."""
        return self.indexing_integration.get_index_stats()

    async def update_metadata_index(self, file_paths: list[str]) -> bool:
        """Update the metadata index for specific files."""
        return await self.indexing_integration.update_index(file_paths)

    async def remove_from_metadata_index(self, file_paths: list[str]) -> bool:
        """Remove files from the metadata index."""
        return await self.indexing_integration.remove_from_index(file_paths)

    async def optimize_metadata_index(self) -> bool:
        """Optimize the metadata index for better performance."""
        return await self.indexing_integration.optimize_index()

    async def clear_metadata_index(self) -> bool:
        """Clear the entire metadata index."""
        return await self.indexing_integration.clear_index()

    def get_metadata_index_size(self) -> dict[str, Any]:
        """Get information about index size and storage."""
        return self.indexing_integration.get_index_size()

    async def backup_metadata_index(self, backup_path: str) -> bool:
        """Create a backup of the metadata index."""
        return await self.indexing_integration.backup_index(backup_path)

    async def restore_metadata_index(self, backup_path: str) -> bool:
        """Restore the metadata index from a backup."""
        return await self.indexing_integration.restore_index(backup_path)

    def get_metadata_index_health(self) -> dict[str, Any]:
        """Get health status of the metadata index."""
        return self.indexing_integration.get_index_health()

    # ── Parallel processing: expose utility methods ──

    def get_optimal_worker_count(self, file_count: int, query: Query) -> int:
        """Calculate optimal worker count based on workload characteristics."""
        return self.parallel_processing.get_optimal_worker_count(file_count, query)

    def should_use_process_pool(self, file_count: int, query: Query) -> bool:
        """Determine if process pool should be used over thread pool."""
        return self.parallel_processing.should_use_process_pool(file_count, query)

    def estimate_search_time(self, file_count: int, query: Query) -> float:
        """
        Estimate search time based on file count and query complexity.

        Args:
            file_count: Number of files to search
            query: Search query

        Returns:
            Estimated search time in milliseconds
        """
        return self.parallel_processing.estimate_search_time(file_count, query)

    # ── History: expose bookmark search/stats, session analytics, performance insights ──

    def search_bookmarks(self, pattern: str) -> list[tuple[str, SearchHistoryEntry]]:
        """
        Search bookmarks by name or query pattern.

        Args:
            pattern: Pattern to search for in bookmark names and query patterns

        Returns:
            List of (name, entry) tuples matching the pattern
        """
        self.history._ensure_managers_loaded()
        if self.history._bookmark_manager:
            return self.history._bookmark_manager.search_bookmarks(pattern)
        return []

    def get_bookmark_stats(self) -> dict[str, Any]:
        """
        Get bookmark statistics.

        Returns:
            Dictionary with total bookmarks, total folders, and bookmarks in folders
        """
        self.history._ensure_managers_loaded()
        if self.history._bookmark_manager:
            return self.history._bookmark_manager.get_bookmark_stats()
        return {}

    def get_session_analytics(self, days: int = 30) -> dict[str, Any]:
        """
        Get session-based analytics.

        Args:
            days: Number of days to include in analytics

        Returns:
            Dictionary with session analytics data
        """
        self.history._ensure_managers_loaded()
        if self.history._session_manager:
            return self.history._session_manager.get_session_analytics(days)
        return {}

    def get_session_by_id(self, session_id: str) -> Any:
        """
        Get a specific session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SearchSession if found, None otherwise
        """
        self.history._ensure_managers_loaded()
        if self.history._session_manager:
            return self.history._session_manager.get_session_by_id(session_id)
        return None

    def cleanup_old_sessions(self, days: int = 90) -> int:
        """
        Remove sessions older than specified days.

        Args:
            days: Number of days to keep sessions for

        Returns:
            Number of sessions removed
        """
        self.history._ensure_managers_loaded()
        if self.history._session_manager:
            return self.history._session_manager.cleanup_old_sessions(days)
        return 0

    def get_performance_insights(self) -> dict[str, Any]:
        """
        Get performance insights and recommendations based on search history.

        Returns:
            Dictionary with insights, recommendations, and metrics
        """
        self.history._ensure_managers_loaded()
        if self.history._analytics_manager:
            return self.history._analytics_manager.get_performance_insights(
                list(self.history._history)
            )
        return {}

    def get_usage_patterns(self) -> dict[str, Any]:
        """
        Analyze usage patterns and trends from search history.

        Returns:
            Dictionary with temporal patterns, search patterns, and productivity metrics
        """
        self.history._ensure_managers_loaded()
        if self.history._analytics_manager:
            return self.history._analytics_manager.get_usage_patterns(
                list(self.history._history)
            )
        return {}

    # ── Multi-Provider Vector Management ──

    async def create_multi_provider_vector_manager(
        self,
        providers: list[str] | None = None,
    ) -> MultiProviderVectorManager:
        """
        Create a multi-provider vector manager for redundancy and performance.

        This allows indexing and searching across multiple vector databases
        simultaneously (e.g., LanceDB + Qdrant) for fault tolerance and
        optimized retrieval.

        Args:
            providers: List of provider types to initialize.
                Supported: "lancedb", "qdrant", "chroma".
                Defaults to ["lancedb"] if not specified.

        Returns:
            Configured MultiProviderVectorManager instance.

        Example:
            >>> manager = await engine.create_multi_provider_vector_manager(
            ...     providers=["lancedb", "qdrant"]
            ... )
        """
        embedding_config = EmbeddingConfig(
            provider=self.cfg.embedding_provider,
            model_name=self.cfg.embedding_model,
            batch_size=self.cfg.embedding_batch_size,
            api_key=self.cfg.embedding_api_key,
        )

        cache_dir = self.cfg.resolve_cache_dir() / "multi_vectors"
        manager = MultiProviderVectorManager(cache_dir, embedding_config)

        providers = providers or ["lancedb"]
        for provider_type in providers:
            await manager.add_provider(provider_type, provider_type)

        self.logger.info(f"Multi-provider vector manager created with providers: {providers}")
        return manager

    # ── GraphRAG Extended Methods ──

    async def reset_knowledge_graph(self) -> bool:
        """
        Reset the knowledge graph and delete all associated vector data.

        This completely clears the knowledge graph, removes cached graph files,
        and deletes the vector collection in Qdrant.

        Returns:
            True if reset was successful, False otherwise.

        Example:
            >>> success = await engine.reset_knowledge_graph()
            >>> if success:
            ...     print("Knowledge graph reset - ready for rebuild")
        """
        return await self.graphrag_integration.reset_graph()

    async def batch_find_similar_entities(
        self, entity_ids: list[str], limit: int = 10
    ) -> dict[str, list[Any]]:
        """
        Find similar entities for multiple entities using batch vector search.

        Uses Qdrant's batch_search API for efficient multi-query similarity
        search, significantly faster than sequential find_similar_entities calls.

        Args:
            entity_ids: List of entity IDs to find similar entities for.
            limit: Maximum number of similar entities per query.

        Returns:
            Dict mapping entity_id to list of similar entities.

        Example:
            >>> results = await engine.batch_find_similar_entities(
            ...     ["entity_1", "entity_2", "entity_3"], limit=5
            ... )
            >>> for eid, similar in results.items():
            ...     print(f"{eid}: {len(similar)} similar entities")
        """
        return await self.graphrag_integration.batch_find_similar(entity_ids, limit)

    async def get_graphrag_stats_async(self) -> dict[str, Any]:
        """
        Get comprehensive GraphRAG statistics including vector store details.

        Returns async stats with vector collection listing and collection info
        from the Qdrant vector store.

        Returns:
            Dictionary with graph stats, vector collections, and collection info.

        Example:
            >>> stats = await engine.get_graphrag_stats_async()
            >>> print(f"Entities: {stats.get('total_entities', 0)}")
            >>> print(f"Collections: {stats.get('vector_collections', [])}")
        """
        return await self.graphrag_integration.get_graph_stats_async()
