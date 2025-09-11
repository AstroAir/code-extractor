"""
Advanced search capabilities including semantic and hybrid search.

This module provides sophisticated search functionality beyond basic text and regex
matching, including semantic similarity search and hybrid search strategies.

Classes:
    AdvancedSearchManager: Manages advanced search capabilities

Key Features:
    - Semantic search with embedding-based similarity
    - Hybrid search combining multiple search methods
    - Code structure awareness and contextual understanding
    - Multi-modal scoring and ranking
    - Advanced result filtering and clustering

Example:
    Using advanced search:
        >>> from pysearch.core.integrations.advanced_search import AdvancedSearchManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = AdvancedSearchManager(config)
        >>> results = manager.search_semantic("database connection", threshold=0.2)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import SearchConfig
from ..types import SearchResult, SearchStats


class AdvancedSearchManager:
    """Manages advanced search capabilities including semantic and hybrid search."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.semantic_engine = None
        self._error_collector = None
        self._logger = None

    def _ensure_semantic_engine(self) -> None:
        """Lazy load the semantic search engine to avoid circular imports."""
        if self.semantic_engine is None:
            from ...search.semantic_advanced import SemanticSearchEngine
            self.semantic_engine = SemanticSearchEngine()

    def set_dependencies(self, error_collector: Any, logger: Any) -> None:
        """Set dependencies for error handling and logging."""
        self._error_collector = error_collector
        self._logger = logger

    def search_semantic_advanced(
        self,
        query: str,
        file_paths: list[Path],
        get_file_content_func: callable,
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
            file_paths: List of file paths to search
            get_file_content_func: Function to get file content
            threshold: Minimum semantic similarity threshold (0.0-1.0)
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            SearchResult with semantically relevant matches
        """
        start_time = time.time()

        # Clear previous errors
        if self._error_collector:
            self._error_collector.clear()

        # Log search start
        if self._logger:
            self._logger.log_search_start(
                pattern=f"semantic:{query}",
                paths=self.config.paths,
                use_regex=False,
                use_ast=False,
            )

        self._ensure_semantic_engine()

        # Prepare semantic engine by fitting on corpus if needed
        if not self.semantic_engine.embedding_model.is_fitted and file_paths:
            # Sample documents for training (limit to avoid memory issues)
            sample_docs = []
            for path in file_paths[:1000]:  # Limit to 1000 files for training
                try:
                    content = get_file_content_func(path)
                    if content:
                        sample_docs.append(content)
                except Exception:
                    continue

            if sample_docs:
                self.semantic_engine.fit_corpus(sample_docs)

        # Perform semantic search on all files
        all_semantic_matches = []
        files_processed = 0

        for path in file_paths:
            if files_processed >= max_results * 2:  # Process more files than needed
                break

            try:
                content = get_file_content_func(path)
                if not content:
                    continue

                # Apply metadata filters if specified
                metadata_filters = kwargs.get('metadata_filters')
                if metadata_filters:
                    from ...utils.utils import create_file_metadata
                    from ...utils.metadata_filters import apply_metadata_filters
                    
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
                if self._error_collector:
                    from ...utils.error_handling import handle_file_error
                    handle_file_error(path, "semantic search", e, self._error_collector)
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
            indexed_files=len(file_paths),
        )

        result = SearchResult(items=search_items, stats=stats)

        # Log completion
        if self._logger:
            self._logger.log_search_complete(
                pattern=f"semantic:{query}",
                results_count=len(search_items),
                elapsed_ms=elapsed_ms
            )

        return result

    async def hybrid_search(
        self,
        pattern: str,
        traditional_search_func: callable,
        graphrag_search_func: callable | None = None,
        enhanced_index_search_func: callable | None = None,
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
            traditional_search_func: Function for traditional search
            graphrag_search_func: Function for GraphRAG search (optional)
            enhanced_index_search_func: Function for enhanced index search (optional)
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
            from ..types import Query
            traditional_query = Query(pattern=pattern, **kwargs)
            traditional_results = traditional_search_func(traditional_query)
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
            if self._logger:
                self._logger.error(f"Traditional search failed: {e}")

        # GraphRAG search
        if use_graphrag and graphrag_search_func:
            try:
                from ..types import GraphRAGQuery
                graphrag_query = GraphRAGQuery(
                    pattern=pattern,
                    max_hops=graphrag_max_hops,
                    include_relationships=True
                )
                graphrag_results = await graphrag_search_func(graphrag_query)
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
                if self._logger:
                    self._logger.error(f"GraphRAG search failed: {e}")

        # Enhanced index search
        if use_enhanced_index and enhanced_index_search_func:
            try:
                from ...indexing.metadata import IndexQuery
                index_query = IndexQuery(
                    semantic_query=pattern,
                    include_entities=True,
                    limit=50
                )
                index_results = await enhanced_index_search_func(index_query)
                if index_results:
                    results["enhanced_index"] = index_results
                    results["metadata"]["methods_used"].append("enhanced_index")
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Enhanced index search failed: {e}")

        return results

    def cluster_results_by_similarity(self, search_items: list, similarity_threshold: float = 0.8) -> list:
        """
        Cluster search results by similarity to reduce redundancy.

        Args:
            search_items: List of search items to cluster
            similarity_threshold: Threshold for considering items similar

        Returns:
            List of clustered search items
        """
        try:
            from ...search.scorer import cluster_results_by_similarity
            return cluster_results_by_similarity(search_items, similarity_threshold)
        except Exception:
            return search_items

    def deduplicate_results(self, search_items: list) -> list:
        """
        Remove overlapping or duplicate search results.

        Args:
            search_items: List of search items to deduplicate

        Returns:
            List of deduplicated search items
        """
        try:
            from ...search.scorer import deduplicate_overlapping_results
            return deduplicate_overlapping_results(search_items)
        except Exception:
            return search_items

    def rank_results(self, search_items: list, pattern: str, strategy: str = "hybrid") -> list:
        """
        Rank search results using specified strategy.

        Args:
            search_items: List of search items to rank
            pattern: Original search pattern
            strategy: Ranking strategy to use

        Returns:
            List of ranked search items
        """
        try:
            from ...search.scorer import sort_items, RankingStrategy
            
            # Convert strategy string to enum
            ranking_strategy = getattr(RankingStrategy, strategy.upper(), RankingStrategy.HYBRID)
            
            return sort_items(search_items, self.config, pattern, ranking_strategy)
        except Exception:
            return search_items
