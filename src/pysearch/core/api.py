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

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from ..indexing.cache_manager import CacheManager
from .config import SearchConfig
from ..analysis.dependency_analysis import DependencyAnalyzer, DependencyGraph, DependencyMetrics
from ..indexing.metadata import MetadataIndexer, IndexQuery
from ..utils.error_handling import ErrorCollector, create_error_report, handle_file_error
from ..utils.file_watcher import FileEvent, WatchManager
from ..analysis.graphrag.engine import GraphRAGEngine
from .history import SearchHistory, SearchHistoryEntry
from ..indexing.indexer import Indexer
from ..utils.logging_config import SearchLogger, get_logger
from ..search.matchers import search_in_file
from ..utils.metadata_filters import apply_metadata_filters, get_file_author
from ..integrations.multi_repo import MultiRepoSearchEngine, MultiRepoSearchResult, RepositoryInfo
from ..storage.qdrant_client import QdrantConfig, QdrantVectorStore
from ..search.scorer import (
    RankingStrategy,
    cluster_results_by_similarity,
    deduplicate_overlapping_results,
    sort_items,
)
from ..search.semantic_advanced import SemanticSearchEngine
from .types import (
    GraphRAGQuery, GraphRAGResult, OutputFormat, Query, SearchItem, SearchResult, SearchStats
)
from ..utils.utils import create_file_metadata, read_text_safely


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
        enable_enhanced_indexing: bool = False
    ) -> None:
        """
        Initialize the PySearch engine.

        Args:
            config: Search configuration object. If None, uses default configuration.
            logger: Custom logger instance. If None, uses default logger.
            qdrant_config: Qdrant vector database configuration for GraphRAG.
            enable_graphrag: Whether to enable GraphRAG capabilities.
            enable_enhanced_indexing: Whether to enable enhanced metadata indexing.
        """
        self.cfg = config or SearchConfig()
        self.indexer = Indexer(self.cfg)
        self.history = SearchHistory(self.cfg)
        self.logger = logger or get_logger()
        self.error_collector = ErrorCollector()
        self.semantic_engine = SemanticSearchEngine()
        self.dependency_analyzer = DependencyAnalyzer()
        self.watch_manager = WatchManager()
        self._auto_watch_enabled = False
        self.cache_manager: CacheManager | None = None
        self._caching_enabled = False
        self.multi_repo_engine: MultiRepoSearchEngine | None = None
        self._multi_repo_enabled = False

        # GraphRAG and enhanced indexing components
        self.qdrant_config = qdrant_config
        self.enable_graphrag = enable_graphrag
        self.enable_enhanced_indexing = enable_enhanced_indexing
        self._graphrag_engine: GraphRAGEngine | None = None
        self._enhanced_indexer: MetadataIndexer | None = None
        self._vector_store: QdrantVectorStore | None = None
        self._graphrag_initialized = False
        self._enhanced_indexing_initialized = False

        # In-memory caches
        self._file_content_cache: dict[Path, tuple[float, str]] = {}
        self._search_result_cache: dict[str, tuple[float, SearchResult]] = {}
        self._cache_lock = threading.RLock()
        self.cache_ttl = 300  # 5 minutes TTL

    async def initialize_graphrag(self) -> None:
        """Initialize GraphRAG components."""
        if not self.enable_graphrag or self._graphrag_initialized:
            return

        try:
            if not self._graphrag_engine:
                self._graphrag_engine = GraphRAGEngine(
                    self.cfg, self.qdrant_config)

            await self._graphrag_engine.initialize()
            # Set the vector store reference from the GraphRAG engine
            self._vector_store = self._graphrag_engine.vector_store
            self._graphrag_initialized = True
            self.logger.info("GraphRAG engine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize GraphRAG: {e}")
            self.error_collector.add_error(e)

    async def initialize_enhanced_indexing(self) -> None:
        """Initialize enhanced indexing components."""
        if not self.enable_enhanced_indexing or self._enhanced_indexing_initialized:
            return

        try:
            if not self._enhanced_indexer:
                self._enhanced_indexer = MetadataIndexer(self.cfg)

            await self._enhanced_indexer.initialize()
            self._enhanced_indexing_initialized = True
            self.logger.info("Enhanced indexing initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced indexing: {e}")
            self.error_collector.add_error(e)

    async def build_knowledge_graph(self, force_rebuild: bool = False) -> bool:
        """Build the GraphRAG knowledge graph."""
        if not self.enable_graphrag:
            self.logger.warning("GraphRAG not enabled")
            return False

        try:
            await self.initialize_graphrag()
            if self._graphrag_engine:
                await self._graphrag_engine.build_knowledge_graph(force_rebuild)
                self.logger.info("Knowledge graph built successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to build knowledge graph: {e}")
            self.error_collector.add_error(e)

        return False

    async def build_enhanced_index(self, include_semantic: bool = True, force_rebuild: bool = False) -> bool:
        """Build the enhanced metadata index."""
        if not self.enable_enhanced_indexing:
            self.logger.warning("Enhanced indexing not enabled")
            return False

        try:
            await self.initialize_enhanced_indexing()
            if self._enhanced_indexer:
                stats = await self._enhanced_indexer.build_index(include_semantic, force_rebuild)
                self.logger.info(
                    f"Enhanced index built: {stats.total_files} files, {stats.total_entities} entities")
                return True
        except Exception as e:
            self.logger.error(f"Failed to build enhanced index: {e}")
            self.error_collector.add_error(e)

        return False

    async def graphrag_search(self, query: GraphRAGQuery) -> GraphRAGResult | None:
        """Perform GraphRAG-based search."""
        if not self.enable_graphrag:
            self.logger.warning("GraphRAG not enabled")
            return None

        try:
            await self.initialize_graphrag()
            if self._graphrag_engine:
                result = await self._graphrag_engine.query_graph(query)
                self.logger.debug(
                    f"GraphRAG search found {len(result.entities)} entities")
                return result
        except Exception as e:
            self.logger.error(f"GraphRAG search failed: {e}")
            self.error_collector.add_error(e)

        return None

    async def enhanced_index_search(self, query: IndexQuery) -> dict[str, Any] | None:
        """Search using the enhanced metadata index."""
        if not self.enable_enhanced_indexing:
            self.logger.warning("Enhanced indexing not enabled")
            return None

        try:
            await self.initialize_enhanced_indexing()
            if self._enhanced_indexer:
                results = await self._enhanced_indexer.query_index(query)
                self.logger.debug(
                    f"Enhanced index search found {len(results.get('files', []))} files")
                return results
        except Exception as e:
            self.logger.error(f"Enhanced index search failed: {e}")
            self.error_collector.add_error(e)

        return None

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

        return search_in_file(path, text, query)

    def _get_cached_file_content(self, path: Path) -> str:
        """
        Get file content with caching based on modification time.

        This method implements an in-memory cache for file contents to avoid
        repeatedly reading the same files. Cache entries are invalidated when
        the file's modification time changes.

        Args:
            path: Path to the file to read

        Returns:
            File content as string if successful, None if file cannot be read
            or exceeds size limits

        Note:
            The cache is automatically pruned when it exceeds 1000 entries to
            prevent excessive memory usage.
        """
        try:
            stat = path.stat()
            current_mtime = stat.st_mtime

            with self._cache_lock:
                if path in self._file_content_cache:
                    cached_mtime, content = self._file_content_cache[path]
                    if cached_mtime == current_mtime:
                        return content

                # Cache miss or outdated - read file
                try:
                    file_content = read_text_safely(
                        path, max_bytes=self.cfg.max_file_bytes)
                    if file_content is not None:
                        self._file_content_cache[path] = (
                            current_mtime, file_content)
                        return file_content
                    else:
                        self.logger.debug(f"Could not read file: {path}")
                        return ""
                except Exception as e:
                    handle_file_error(
                        path, "read", e, self.error_collector, self.logger)
                    return ""
                    # Limit cache size
                    if len(self._file_content_cache) > 1000:
                        # Remove oldest 20% of entries
                        to_remove = list(self._file_content_cache.keys())[:200]
                        for k in to_remove:
                            del self._file_content_cache[k]

                return content
        except Exception:
            return ""

    def _get_cache_key(self, query: Query) -> str:
        """Generate cache key for search query."""
        return f"{query.pattern}:{query.use_regex}:{query.use_ast}:{query.context}:{hash(str(query.filters))}:{hash(str(query.metadata_filters))}"

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl

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
        if use_cache and self._caching_enabled and self.cache_manager:
            cache_key = self._generate_cache_key(query)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.logger.debug(
                    f"Using cached result for query: {query.pattern}")
                return cached_result

        # Fallback to old cache system
        cache_key = self._get_cache_key(query)
        with self._cache_lock:
            if cache_key in self._search_result_cache:
                timestamp, result = self._search_result_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    self.logger.debug("Returning cached search result")
                    return result

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
            _removed: list[str] = []
            total_seen = 0

        paths = changed or list(self.indexer.iter_all_paths())
        self.logger.debug(f"Searching in {len(paths)} files")

        # Adaptive parallelization based on workload
        try:
            items = self._search_with_adaptive_parallelism(paths, query)
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            self.error_collector.add_error(e)
            items = []

        # Sort results by relevance with configurable strategy
        try:
            # Use hybrid ranking by default, could be configurable
            items = sort_items(items, self.cfg, query.pattern,
                               RankingStrategy.HYBRID)
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

        # Cache result in new cache manager if enabled
        if use_cache and self._caching_enabled and self.cache_manager:
            cache_key = self._generate_cache_key(query)
            file_dependencies = self._get_file_dependencies(result)
            self.cache_manager.set(
                key=cache_key,
                value=result,
                file_dependencies=file_dependencies
            )

        # Cache result in old cache system
        with self._cache_lock:
            self._search_result_cache[cache_key] = (time.time(), result)
            # Limit cache size
            if len(self._search_result_cache) > 100:
                oldest_keys = sorted(
                    self._search_result_cache.keys(), key=lambda k: self._search_result_cache[k][0]
                )[:20]
                for k in oldest_keys:
                    del self._search_result_cache[k]

        return result

    def _search_with_adaptive_parallelism(
        self, paths: list[Path], query: Query
    ) -> list[SearchItem]:
        """Adaptive parallelization strategy based on workload size and complexity."""
        items: list[SearchItem] = []
        num_files = len(paths)

        if not self.cfg.parallel or num_files < 10:
            # Sequential for small workloads
            for p in paths:
                res = self._search_file(p, query)
                if res:
                    items.extend(res)
            return items

        # Choose between thread and process pool based on workload
        cpu_count = os.cpu_count() or 4

        if num_files > 1000 and not query.use_ast:  # Heavy I/O workload
            # Use process pool for CPU-intensive work with many files
            workers = min(cpu_count, self.cfg.workers or cpu_count)
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    # Batch files to reduce overhead
                    batch_size = max(1, num_files // (workers * 4))
                    batches = [paths[i: i + batch_size]
                               for i in range(0, len(paths), batch_size)]

                    futures = {
                        executor.submit(_search_file_batch, batch, query): batch
                        for batch in batches
                    }

                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            for result in batch_results:
                                if result:
                                    items.extend(result)
                        except Exception:
                            # Fall back to thread pool on process pool failure
                            batch = futures[future]
                            for p in batch:
                                res = self._search_file(p, query)
                                if res:
                                    items.extend(res)
            except Exception:
                # Fallback to thread pool
                items = self._search_with_thread_pool(paths, query)
        else:
            # Use thread pool for I/O bound or smaller workloads
            items = self._search_with_thread_pool(paths, query)

        return items

    def _search_with_thread_pool(self, paths: list[Path], query: Query) -> list[SearchItem]:
        """Thread-based parallel search with optimized worker count."""
        items: list[SearchItem] = []
        cpu_count = os.cpu_count() or 4

        # Optimize worker count based on I/O vs CPU ratio
        if query.use_ast or query.use_regex:
            workers = min(cpu_count * 2, self.cfg.workers or cpu_count * 2)
        else:
            # I/O bound
            workers = min(cpu_count * 4, self.cfg.workers or cpu_count * 4)

        workers = min(workers, len(paths))  # Don't over-provision

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(
                self._search_file, p, query): p for p in paths}

            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        items.extend(res)
                except Exception:
                    # Log error but continue processing
                    pass

        return items

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        with self._cache_lock:
            self._file_content_cache.clear()
            self._search_result_cache.clear()

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

    async def hybrid_search(
        self,
        pattern: str,
        use_graphrag: bool = True,
        use_enhanced_index: bool = True,
        graphrag_max_hops: int = 2,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Perform hybrid search combining traditional search, GraphRAG, and enhanced indexing.

        This method provides a unified interface that leverages all available search
        capabilities to provide comprehensive results.

        Args:
            pattern: Search pattern or query
            use_graphrag: Whether to include GraphRAG results
            use_enhanced_index: Whether to include enhanced index results
            graphrag_max_hops: Maximum hops for GraphRAG traversal
            **kwargs: Additional parameters for traditional search

        Returns:
            Dictionary containing results from all enabled search methods
        """
        results: dict[str, Any] = {
            "traditional": None,
            "graphrag": None,
            "enhanced_index": None,
            "metadata": {
                "pattern": pattern,
                "timestamp": time.time(),
                "methods_used": []
            }
        }

        # Traditional search
        try:
            traditional_query = Query(pattern=pattern, **kwargs)
            traditional_results = self.run(traditional_query)
            results["traditional"] = {
                "items": [
                    {
                        "file": str(item.file),
                        "start_line": item.start_line,
                        "end_line": item.end_line,
                        "lines": item.lines,
                        "match_spans": item.match_spans
                    }
                    for item in traditional_results.items
                ],
                "stats": {
                    "files_scanned": traditional_results.stats.files_scanned,
                    "files_matched": traditional_results.stats.files_matched,
                    "items": traditional_results.stats.items,
                    "elapsed_ms": traditional_results.stats.elapsed_ms
                }
            }
            results["metadata"]["methods_used"].append("traditional")
        except Exception as e:
            self.logger.error(f"Traditional search failed: {e}")

        # GraphRAG search
        if use_graphrag and self.enable_graphrag:
            try:
                graphrag_query = GraphRAGQuery(
                    pattern=pattern,
                    max_hops=graphrag_max_hops,
                    include_relationships=True
                )
                graphrag_results = await self.graphrag_search(graphrag_query)
                if graphrag_results:
                    results["graphrag"] = {
                        "entities": [
                            {
                                "id": entity.id,
                                "name": entity.name,
                                "type": entity.entity_type.value,
                                "file": str(entity.file_path),
                                "line": entity.start_line,
                                "signature": entity.signature,
                                "docstring": entity.docstring
                            }
                            for entity in graphrag_results.entities
                        ],
                        "relationships": [
                            {
                                "source": rel.source_entity_id,
                                "target": rel.target_entity_id,
                                "type": rel.relation_type.value,
                                "confidence": rel.confidence,
                                "context": rel.context
                            }
                            for rel in graphrag_results.relationships
                        ],
                        "similarity_scores": graphrag_results.similarity_scores,
                        "metadata": graphrag_results.metadata
                    }
                    results["metadata"]["methods_used"].append("graphrag")
            except Exception as e:
                self.logger.error(f"GraphRAG search failed: {e}")

        # Enhanced index search
        if use_enhanced_index and self.enable_enhanced_indexing:
            try:
                index_query = IndexQuery(
                    semantic_query=pattern,
                    include_entities=True,
                    limit=50
                )
                index_results = await self.enhanced_index_search(index_query)
                if index_results:
                    results["enhanced_index"] = index_results
                    results["metadata"]["methods_used"].append(
                        "enhanced_index")
            except Exception as e:
                self.logger.error(f"Enhanced index search failed: {e}")

        return results

    async def close_async_components(self) -> None:
        """Close async components properly."""
        if self._graphrag_engine:
            await self._graphrag_engine.close()

        if self._enhanced_indexer:
            await self._enhanced_indexer.close()

        if self._vector_store:
            await self._vector_store.close()

    def search_semantic_advanced(
        self,
        query: str,
        threshold: float = 0.1,
        max_results: int = 100,
        **kwargs: Any
    ) -> SearchResult:
        """
        Perform advanced semantic search with embedding-based similarity.

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
            >>> results = engine.search_semantic_advanced("database connection")
            >>>
            >>> # Find web API implementations
            >>> results = engine.search_semantic_advanced("web api", threshold=0.2)
            >>>
            >>> # Find testing utilities
            >>> results = engine.search_semantic_advanced("test utilities", max_results=50)
        """
        start_time = time.time()

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
                f"Indexer scan: {len(changed)} changed, {len(removed)} removed, {total_seen} total")
        except Exception as e:
            self.error_collector.add_error(e)
            # Continue with empty file list
            changed, removed, total_seen = [], [], 0

        paths = changed or list(self.indexer.iter_all_paths())

        # Prepare semantic engine by fitting on corpus if needed
        if not self.semantic_engine.embedding_model.is_fitted and paths:
            # Sample documents for training (limit to avoid memory issues)
            sample_docs = []
            for path in paths[:1000]:  # Limit to 1000 files for training
                try:
                    content = self._get_cached_file_content(path)
                    if content:
                        sample_docs.append(content)
                except Exception:
                    continue

            if sample_docs:
                self.semantic_engine.fit_corpus(sample_docs)

        # Perform semantic search on all files
        all_semantic_matches = []
        files_processed = 0

        for path in paths:
            if files_processed >= max_results * 2:  # Process more files than needed
                break

            try:
                content = self._get_cached_file_content(path)
                if not content:
                    continue

                # Apply metadata filters if specified
                metadata_filters = kwargs.get('metadata_filters')
                if metadata_filters:
                    file_metadata = create_file_metadata(path)
                    if file_metadata and not apply_metadata_filters(file_metadata, metadata_filters):
                        continue

                # Perform semantic search on this file
                semantic_matches = self.semantic_engine.search_semantic(
                    query=query,
                    content=content,
                    file_path=path,
                    threshold=threshold
                )

                all_semantic_matches.extend(semantic_matches)
                files_processed += 1

            except Exception as e:
                handle_file_error(path, "semantic search",
                                  e, self.error_collector)
                continue

        # Convert semantic matches to SearchItems
        search_items = []
        for semantic_match in all_semantic_matches:
            search_items.append(semantic_match.item)

        # Sort by semantic relevance (combined score)
        search_items.sort(
            key=lambda item: next(
                (m.combined_score for m in all_semantic_matches if m.item == item), 0.0
            ),
            reverse=True
        )

        # Limit results
        search_items = search_items[:max_results]

        # Calculate statistics
        elapsed_ms = (time.time() - start_time) * 1000
        stats = SearchStats(
            files_scanned=files_processed,
            files_matched=len(set(item.file for item in search_items)),
            items=len(search_items),
            elapsed_ms=elapsed_ms,
            indexed_files=total_seen,
        )

        result = SearchResult(items=search_items, stats=stats)

        # Log completion
        self.logger.log_search_complete(
            pattern=f"semantic:{query}",
            results_count=len(search_items),
            elapsed_ms=elapsed_ms
        )

        # Add to history
        semantic_query = Query(
            pattern=f"semantic:{query}",
            use_regex=False,
            use_semantic=True,
            filters=None,
            metadata_filters=None
        )
        self.history.add_search(semantic_query, result)

        return result

    def analyze_dependencies(
        self,
        directory: Path | None = None,
        recursive: bool = True
    ) -> DependencyGraph:
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
        if directory is None:
            if self.cfg.paths:
                directory = Path(self.cfg.paths[0])
            else:
                directory = Path.cwd()

        self.logger.info(f"Starting dependency analysis for: {directory}")
        start_time = time.time()

        # Perform dependency analysis
        graph = self.dependency_analyzer.analyze_directory(
            directory, recursive)

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"Dependency analysis completed: {len(graph.nodes)} modules, "
            f"{sum(len(edges) for edges in graph.edges.values())} dependencies, "
            f"time={elapsed_ms:.2f}ms"
        )

        return graph

    def get_dependency_metrics(self, graph: DependencyGraph | None = None) -> DependencyMetrics:
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
        if graph is None:
            graph = self.analyze_dependencies()

        # Update analyzer's graph and calculate metrics
        self.dependency_analyzer.graph = graph
        return self.dependency_analyzer.calculate_metrics()

    def find_dependency_impact(
        self,
        module: str,
        graph: DependencyGraph | None = None
    ) -> dict[str, Any]:
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
        if graph is None:
            graph = self.analyze_dependencies()

        # Update analyzer's graph and perform impact analysis
        self.dependency_analyzer.graph = graph
        return self.dependency_analyzer.find_impact_analysis(module)

    def suggest_refactoring_opportunities(
        self,
        graph: DependencyGraph | None = None
    ) -> list[dict[str, Any]]:
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
        if graph is None:
            graph = self.analyze_dependencies()

        # Update analyzer's graph and get suggestions
        self.dependency_analyzer.graph = graph
        return self.dependency_analyzer.suggest_refactoring_opportunities()

    def enable_auto_watch(
        self,
        debounce_delay: float = 0.5,
        batch_size: int = 50,
        max_batch_delay: float = 5.0
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
        if self._auto_watch_enabled:
            self.logger.warning("Auto-watch already enabled")
            return True

        try:
            # Create watchers for all configured paths
            for i, path in enumerate(self.cfg.paths):
                watcher_name = f"path_{i}"
                success = self.watch_manager.add_watcher(
                    name=watcher_name,
                    path=path,
                    config=self.cfg,
                    indexer=self.indexer,
                    debounce_delay=debounce_delay,
                    batch_size=batch_size,
                    max_batch_delay=max_batch_delay
                )

                if not success:
                    self.logger.error(
                        f"Failed to create watcher for path: {path}")
                    return False

            # Start all watchers
            started = self.watch_manager.start_all()
            if started == len(self.cfg.paths):
                self._auto_watch_enabled = True
                self.logger.info(f"Auto-watch enabled for {started} paths")
                return True
            else:
                self.logger.error(
                    f"Only {started}/{len(self.cfg.paths)} watchers started")
                self.watch_manager.stop_all()
                return False

        except Exception as e:
            self.logger.error(f"Failed to enable auto-watch: {e}")
            return False

    def disable_auto_watch(self) -> None:
        """
        Disable automatic file watching.

        Stops all file watchers and disables real-time index updates.
        The search index will need to be manually refreshed after this.

        Example:
            >>> engine.disable_auto_watch()
            >>> print("Auto-watch disabled - manual index refresh required")
        """
        if not self._auto_watch_enabled:
            return

        try:
            self.watch_manager.stop_all()
            self._auto_watch_enabled = False
            self.logger.info("Auto-watch disabled")
        except Exception as e:
            self.logger.error(f"Error disabling auto-watch: {e}")

    def is_auto_watch_enabled(self) -> bool:
        """
        Check if automatic file watching is enabled.

        Returns:
            True if auto-watch is enabled, False otherwise
        """
        return self._auto_watch_enabled

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
        return self.watch_manager.get_all_stats()

    def add_custom_watcher(
        self,
        name: str,
        path: Path | str,
        change_handler: Callable[[list[FileEvent]], None],
        **kwargs: Any
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
        return self.watch_manager.add_watcher(
            name=name,
            path=path,
            config=self.cfg,
            change_handler=change_handler,
            **kwargs
        )

    def remove_watcher(self, name: str) -> bool:
        """
        Remove a file watcher by name.

        Args:
            name: Name of the watcher to remove

        Returns:
            True if watcher was removed successfully, False otherwise
        """
        return self.watch_manager.remove_watcher(name)

    def list_watchers(self) -> list[str]:
        """
        Get list of active file watcher names.

        Returns:
            List of watcher names
        """
        return self.watch_manager.list_watchers()

    def enable_caching(
        self,
        backend: str = "memory",
        cache_dir: Path | str | None = None,
        max_size: int = 1000,
        default_ttl: float = 3600,
        compression: bool = False
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
                compression=compression
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
            return self.cache_manager.get_stats()
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
            return self.cache_manager.invalidate_by_file(str(file_path))
        return 0

    def _generate_cache_key(self, query: Query) -> str:
        """Generate a cache key for a search query."""
        # Create a deterministic key from query parameters
        key_parts = [
            f"pattern:{query.pattern}",
            f"regex:{query.use_regex}",
            f"ast:{query.use_ast}",
            f"semantic:{query.use_semantic}",
            f"context:{query.context}",
        ]

        if query.filters:
            key_parts.append(f"filters:{hash(str(query.filters))}")

        if query.metadata_filters:
            key_parts.append(f"metadata:{hash(str(query.metadata_filters))}")

        # Include configuration that affects results
        key_parts.extend([
            f"paths:{hash(tuple(sorted(self.cfg.paths)))}",
            f"include:{hash(tuple(sorted(self.cfg.include or [])))}",
            f"exclude:{hash(tuple(sorted(self.cfg.exclude or [])))}"
        ])

        return "|".join(key_parts)

    def _get_file_dependencies(self, result: SearchResult) -> set[str]:
        """Extract file dependencies from search result."""
        dependencies = set()
        for item in result.items:
            dependencies.add(str(item.file))
        return dependencies

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
            self.multi_repo_engine = MultiRepoSearchEngine(
                max_workers=max_workers)
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
        **metadata: Any
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

        return self.multi_repo_engine.add_repository(
            name=name,
            path=path,
            config=config,
            priority=priority,
            **metadata
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

        return self.multi_repo_engine.remove_repository(name)

    def list_repositories(self) -> list[str]:
        """
        Get list of repository names in multi-repository search.

        Returns:
            List of repository names, empty if multi-repo not enabled
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return []

        return self.multi_repo_engine.list_repositories()

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

        return self.multi_repo_engine.get_repository_info(name)

    def search_all_repositories(
        self,
        pattern: str,
        use_regex: bool = False,
        use_ast: bool = False,
        use_semantic: bool = False,
        context: int = 2,
        max_results: int = 1000,
        aggregate_results: bool = True,
        timeout: float = 30.0
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

        return self.multi_repo_engine.search_all(
            pattern=pattern,
            use_regex=use_regex,
            use_ast=use_ast,
            use_semantic=use_semantic,
            context=context,
            max_results=max_results,
            aggregate_results=aggregate_results,
            timeout=timeout
        )

    def search_specific_repositories(
        self,
        repositories: list[str],
        query: Query,
        max_results: int = 1000,
        aggregate_results: bool = True,
        timeout: float = 30.0
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

        return self.multi_repo_engine.search_repositories(
            repositories=repositories,
            query=query,
            max_results=max_results,
            aggregate_results=aggregate_results,
            timeout=timeout
        )

    def get_multi_repo_health(self) -> dict[str, Any]:
        """
        Get health status for all repositories in multi-repository system.

        Returns:
            Dictionary with health information, empty if not enabled
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return {}

        return self.multi_repo_engine.get_health_status()

    def get_multi_repo_stats(self) -> dict[str, Any]:
        """
        Get search performance statistics for multi-repository system.

        Returns:
            Dictionary with search statistics, empty if not enabled
        """
        if not self._multi_repo_enabled or not self.multi_repo_engine:
            return {}

        return self.multi_repo_engine.get_search_statistics()

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
        from ..search.fuzzy import FuzzyAlgorithm

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

        # For now, use the first algorithm for regex generation
        # In a full implementation, you'd want to process results differently
        if fuzzy_algos:
            from ..search.fuzzy import fuzzy_pattern

            fuzzy_regex = fuzzy_pattern(pattern, max_distance, fuzzy_algos[0])
            return self.search(fuzzy_regex, regex=True, **kwargs)
        else:
            return self.search(pattern, regex=False, **kwargs)

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
        return self.history.get_sessions(limit)

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
            _all_files = {item.file for item in result.items}
            sorted_items = sort_items(
                result.items, self.cfg, pattern, strategy)

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
                analysis["suggestions"].append(
                    "Identifiers work well with hybrid ranking")
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
                    analysis["suggestions"].append(
                        "Many files found - consider clustering results")

                # Calculate result diversity
                if len(results.items) > 1:
                    clusters = cluster_results_by_similarity(results.items)
                    analysis["result_diversity"] = len(
                        clusters) / len(results.items)

                    if analysis["result_diversity"] < 0.3:
                        analysis["suggestions"].append(
                            "Low diversity - results are very similar")
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


# Helper function for process pool (must be at module level)
def _search_file_batch(paths: list[Path], query: Query) -> list[list[SearchItem]]:
    """Process a batch of files in a separate process."""
    from ..search.matchers import search_in_file
    from ..utils.utils import read_text_safely

    results = []
    for path in paths:
        try:
            text = read_text_safely(path)
            if text is not None:
                result = search_in_file(path, text, query)
                results.append(result)
            else:
                results.append([])
        except Exception:
            results.append([])

    return results
