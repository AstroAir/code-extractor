"""
Enhanced indexing integration module.

This module provides the main integration point for the enhanced code indexing
engine, allowing seamless integration with the existing pysearch system while
providing access to all enhanced features.

Classes:
    IndexSearchEngine: Main enhanced search engine
    SearchResultEnhancer: Enhances search results with additional metadata
    IndexingOrchestrator: Orchestrates indexing operations

Features:
    - Seamless integration with existing pysearch APIs
    - Enhanced search results with semantic similarity
    - Automatic index management and optimization
    - Performance monitoring and health checks
    - Backward compatibility with existing code

Example:
    Basic enhanced search:
        >>> from pysearch.enhanced_integration import IndexSearchEngine
        >>> engine = IndexSearchEngine(config)
        >>> results = await engine.search("database connection", limit=10)

    Advanced search with filters:
        >>> results = await engine.index_search(
        ...     query="user authentication",
        ...     languages=["python", "javascript"],
        ...     entity_types=["function", "class"],
        ...     semantic_threshold=0.8
        ... )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...analysis.content_addressing import IndexTag
from ...core.config import SearchConfig
from ...core.types import Language
from ...utils.logging_config import get_logger
from ...utils.performance_monitoring import PerformanceMonitor
from .engine import IndexingEngine

logger = get_logger()


@dataclass
class IndexSearchResult:
    """Enhanced search result with additional metadata."""

    path: str
    content: str
    start_line: int
    end_line: int
    language: Language
    score: float

    # Enhanced metadata
    entity_name: str | None = None
    entity_type: str | None = None
    similarity_score: float | None = None
    complexity_score: float | None = None
    quality_score: float | None = None
    dependencies: list[str] = field(default_factory=list)
    context: str | None = None

    # Source information
    index_type: str = "unknown"
    chunk_id: str | None = None


class SearchResultEnhancer:
    """Enhances search results with additional metadata and context."""

    def __init__(self, config: SearchConfig):
        self.config = config

    async def enhance_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        include_context: bool = True,
    ) -> list[IndexSearchResult]:
        """Enhance search results with additional metadata."""
        enhanced_results = []

        for result in results:
            enhanced_result = IndexSearchResult(
                path=result.get("path", ""),
                content=result.get("content", ""),
                start_line=result.get("start_line", 1),
                end_line=result.get("end_line", 1),
                language=Language(result.get("language", "unknown")),
                score=result.get("score", 0.0),
                entity_name=result.get("entity_name"),
                entity_type=result.get("entity_type"),
                similarity_score=result.get("similarity_score"),
                complexity_score=result.get("complexity_score", 0.0),
                quality_score=result.get("quality_score", 0.0),
                dependencies=result.get("dependencies", []),
                index_type=result.get("index_type", "unknown"),
                chunk_id=result.get("chunk_id"),
            )

            # Add context if requested
            if include_context:
                enhanced_result.context = await self._extract_context(enhanced_result, query)

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def _extract_context(
        self,
        result: IndexSearchResult,
        query: str,
    ) -> str | None:
        """Extract relevant context around the search result."""
        try:
            # Read file and extract context
            file_path = Path(result.path)
            if not file_path.exists():
                return None

            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            # Get context lines around the match
            context_lines = 3
            start_idx = max(0, result.start_line - 1 - context_lines)
            end_idx = min(len(lines), result.end_line + context_lines)

            context_content = "\n".join(lines[start_idx:end_idx])
            return context_content

        except Exception as e:
            logger.debug(f"Error extracting context for {result.path}: {e}")
            return None


class IndexingOrchestrator:
    """Orchestrates indexing operations across the enhanced system."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.indexing_engine: IndexingEngine | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.last_index_time: float | None = None

    async def initialize(self) -> None:
        """Initialize the indexing orchestrator."""
        if self.config.enable_metadata_indexing:
            self.indexing_engine = IndexingEngine(self.config)
            await self.indexing_engine.initialize()

            # Initialize performance monitoring
            cache_dir = self.config.resolve_cache_dir()
            self.performance_monitor = PerformanceMonitor(self.config, cache_dir)
            await self.performance_monitor.start_monitoring()

            logger.info("Enhanced indexing orchestrator initialized")

    async def ensure_indexed(
        self,
        force_refresh: bool = False,
        max_age_hours: float = 24.0,
    ) -> bool:
        """Ensure the codebase is indexed and up-to-date."""
        if not self.indexing_engine:
            return False

        # Check if indexing is needed
        needs_indexing = force_refresh

        if not needs_indexing and self.last_index_time:
            age_hours = (time.time() - self.last_index_time) / 3600
            needs_indexing = age_hours > max_age_hours

        if not needs_indexing:
            # Check if any files have changed
            needs_indexing = await self._check_for_changes()

        if needs_indexing:
            logger.info("Starting indexing operation")

            try:
                async for progress in self.indexing_engine.refresh_index():
                    if progress.progress % 0.2 < 0.01:  # Log every 20%
                        logger.info(f"Indexing progress: {progress.progress:.1%}")

                self.last_index_time = time.time()
                logger.info("Indexing completed successfully")
                return True

            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                return False

        return True

    async def _check_for_changes(self) -> bool:
        """Check if any files have changed since last indexing."""
        try:
            from ..indexer import Indexer

            indexer = Indexer(self.config)
            changed, removed, _total = indexer.scan()
            has_changes = len(changed) > 0 or len(removed) > 0
            if has_changes:
                logger.debug(
                    f"Detected changes: {len(changed)} modified, {len(removed)} removed"
                )
            return has_changes
        except Exception as e:
            logger.error(f"Error checking for changes: {e}")
            return False

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of the indexing system."""
        if not self.performance_monitor:
            return {"status": "disabled"}

        try:
            report = await self.performance_monitor.get_performance_report()
            return {
                "status": "healthy" if report["health_score"] > 0.7 else "degraded",
                "health_score": report["health_score"],
                "last_index_time": self.last_index_time,
                "system_metrics": report.get("system", {}),
            }
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {"status": "error", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()


class IndexSearchEngine:
    """
    Main enhanced search engine that integrates all enhanced features.

    This engine provides a unified interface for enhanced code search,
    combining traditional text search with semantic search, entity extraction,
    and advanced indexing capabilities.
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.orchestrator = IndexingOrchestrator(config)
        self.result_enhancer = SearchResultEnhancer(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the enhanced search engine."""
        if self._initialized:
            return

        await self.orchestrator.initialize()
        self._initialized = True
        logger.info("Enhanced search engine initialized")

    async def search(
        self,
        query: str,
        limit: int = 50,
        include_context: bool = True,
        auto_index: bool = True,
    ) -> list[IndexSearchResult]:
        """
        Perform enhanced search across all available indexes.

        Args:
            query: Search query
            limit: Maximum number of results
            include_context: Whether to include context around matches
            auto_index: Whether to automatically ensure indexes are up-to-date

        Returns:
            List of enhanced search results
        """
        await self.initialize()

        # Ensure indexes are up-to-date
        if auto_index:
            await self.orchestrator.ensure_indexed()

        # Perform search across different index types
        all_results = []

        if self.orchestrator.indexing_engine:
            # Search enhanced indexes
            tag = IndexTag(
                directory=str(Path(self.config.paths[0]).resolve()),
                branch="main",  # TODO: Auto-detect branch
                artifact_id="*",
            )

            # Search code snippets
            try:
                snippets_index = self.orchestrator.indexing_engine.coordinator.get_index(
                    "enhanced_code_snippets"
                )
                if snippets_index:
                    snippet_results = await snippets_index.retrieve(query, tag, limit // 2)
                    for result in snippet_results:
                        result["index_type"] = "code_snippets"
                    all_results.extend(snippet_results)
            except Exception as e:
                logger.debug(f"Code snippets search failed: {e}")

            # Search full-text
            try:
                fulltext_index = self.orchestrator.indexing_engine.coordinator.get_index(
                    "enhanced_full_text"
                )
                if fulltext_index:
                    fulltext_results = await fulltext_index.retrieve(query, tag, limit // 2)
                    for result in fulltext_results:
                        result["index_type"] = "full_text"
                    all_results.extend(fulltext_results)
            except Exception as e:
                logger.debug(f"Full-text search failed: {e}")

            # Search vectors (semantic)
            try:
                vector_index = self.orchestrator.indexing_engine.coordinator.get_index(
                    "enhanced_vectors"
                )
                if vector_index:
                    vector_results = await vector_index.retrieve(query, tag, limit // 2)
                    for result in vector_results:
                        result["index_type"] = "semantic"
                    all_results.extend(vector_results)
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")

        # Fallback to basic search if no enhanced results
        if not all_results:
            logger.info("No enhanced search results, falling back to basic search")
            # TODO: Implement fallback to basic pysearch
            return []

        # Deduplicate and rank results
        deduplicated_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(deduplicated_results, query)

        # Apply vector re-ranking for semantic boost if vector index is available
        try:
            vector_index = self.orchestrator.indexing_engine.coordinator.get_index(
                "enhanced_vectors"
            )
            if vector_index and hasattr(vector_index, "rerank_results"):
                ranked_results = await vector_index.rerank_results(ranked_results, query)
        except Exception as e:
            logger.debug(f"Vector re-ranking skipped: {e}")

        # Limit results
        limited_results = ranked_results[:limit]

        # Enhance results with additional metadata
        enhanced_results = await self.result_enhancer.enhance_results(
            limited_results, query, include_context
        )

        return enhanced_results

    async def index_search(
        self,
        query: str,
        languages: list[str] | None = None,
        entity_types: list[str] | None = None,
        file_patterns: list[str] | None = None,
        semantic_threshold: float = 0.0,
        limit: int = 50,
    ) -> list[IndexSearchResult]:
        """
        Perform advanced search with detailed filtering options.

        Args:
            query: Search query
            languages: Filter by programming languages
            entity_types: Filter by entity types (function, class, etc.)
            file_patterns: Filter by file patterns
            semantic_threshold: Minimum semantic similarity score
            limit: Maximum number of results

        Returns:
            List of enhanced search results
        """
        await self.initialize()

        # Build search filters
        filters: dict[str, Any] = {}
        if languages:
            filters["languages"] = languages
        if entity_types:
            filters["entity_types"] = entity_types
        if file_patterns:
            filters["file_patterns"] = file_patterns
        if semantic_threshold > 0:
            filters["similarity_threshold"] = semantic_threshold

        # Perform filtered search
        # TODO: Implement filtered search logic
        results = await self.search(query, limit)

        # Apply additional filtering
        filtered_results = []
        for result in results:
            # Language filter
            if languages and result.language.value not in languages:
                continue

            # Entity type filter
            if entity_types and result.entity_type not in entity_types:
                continue

            # Semantic threshold filter
            if (
                semantic_threshold > 0
                and result.similarity_score is not None
                and result.similarity_score < semantic_threshold
            ):
                continue

            filtered_results.append(result)

        return filtered_results

    def _deduplicate_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate results based on path and content."""
        seen = set()
        deduplicated = []

        for result in results:
            key = (result.get("path", ""), result.get("start_line", 0))
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)

        return deduplicated

    def _rank_results(
        self,
        results: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Rank results by relevance."""
        # Simple ranking based on multiple factors
        for result in results:
            score = 0.0

            # Base score from index
            if "score" in result:
                score += result["score"] * 0.3

            # Similarity score (for semantic results)
            if "similarity_score" in result and result["similarity_score"]:
                score += result["similarity_score"] * 0.4

            # Quality score
            if "quality_score" in result and result["quality_score"]:
                score += result["quality_score"] * 0.2

            # Exact match bonus
            content = result.get("content", "").lower()
            if query.lower() in content:
                score += 0.1

            result["final_score"] = score

        # Sort by final score
        return sorted(results, key=lambda x: x.get("final_score", 0.0), reverse=True)

    async def get_statistics(self) -> dict[str, Any]:
        """Get search engine statistics."""
        stats: dict[str, Any] = {
            "initialized": self._initialized,
            "metadata_indexing_enabled": self.config.enable_metadata_indexing,
        }

        if self.orchestrator.indexing_engine:
            index_stats = await self.orchestrator.indexing_engine.get_index_stats()
            stats.update(index_stats)

        health_status = await self.orchestrator.get_health_status()
        stats["health"] = health_status

        return stats

    async def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        languages: list[str] | None = None,
        min_quality: float = 0.0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search for code entities (functions, classes, etc.) with advanced filtering.

        Args:
            query: Search query
            entity_types: Filter by entity types
            languages: Filter by programming languages
            min_quality: Minimum quality score
            limit: Maximum number of results

        Returns:
            List of matching entity dictionaries
        """
        await self.initialize()

        if not self.orchestrator.indexing_engine:
            return []

        tag = IndexTag(
            directory=str(Path(self.config.paths[0]).resolve()),
            branch="main",
            artifact_id="enhanced_code_snippets",
        )

        snippets_index = self.orchestrator.indexing_engine.coordinator.get_index(
            "enhanced_code_snippets"
        )
        if snippets_index and hasattr(snippets_index, "search_entities"):
            return await snippets_index.search_entities(
                query, tag, entity_types=entity_types, languages=languages,
                min_quality=min_quality, limit=limit,
            )
        return []

    async def get_entities_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get all code entities for a specific file.

        Args:
            file_path: Path to the source file

        Returns:
            List of entity dictionaries
        """
        await self.initialize()

        if not self.orchestrator.indexing_engine:
            return []

        tag = IndexTag(
            directory=str(Path(self.config.paths[0]).resolve()),
            branch="main",
            artifact_id="enhanced_code_snippets",
        )

        snippets_index = self.orchestrator.indexing_engine.coordinator.get_index(
            "enhanced_code_snippets"
        )
        if snippets_index and hasattr(snippets_index, "get_entities_by_file"):
            return await snippets_index.get_entities_by_file(file_path, tag)
        return []

    async def get_entity_by_id(self, entity_id: int) -> dict[str, Any] | None:
        """Get a specific code entity by its database ID.

        Args:
            entity_id: Entity database ID

        Returns:
            Entity dictionary or None
        """
        await self.initialize()

        if not self.orchestrator.indexing_engine:
            return None

        snippets_index = self.orchestrator.indexing_engine.coordinator.get_index(
            "enhanced_code_snippets"
        )
        if snippets_index and hasattr(snippets_index, "get_entity_by_id"):
            return await snippets_index.get_entity_by_id(entity_id)
        return None

    async def get_chunks_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get all code chunks for a specific file.

        Args:
            file_path: Path to the source file

        Returns:
            List of chunk dictionaries
        """
        await self.initialize()

        if not self.orchestrator.indexing_engine:
            return []

        tag = IndexTag(
            directory=str(Path(self.config.paths[0]).resolve()),
            branch="main",
            artifact_id="enhanced_chunks",
        )

        chunk_index = self.orchestrator.indexing_engine.coordinator.get_index(
            "enhanced_chunks"
        )
        if chunk_index and hasattr(chunk_index, "get_chunks_by_file"):
            return await chunk_index.get_chunks_by_file(file_path, tag)
        return []

    async def search_in_file(
        self,
        file_path: str,
        query: str,
        context_lines: int = 3,
    ) -> list[dict[str, Any]]:
        """Search for a query within a specific indexed file with context.

        Args:
            file_path: Path to the file to search in
            query: Search query
            context_lines: Number of context lines around matches

        Returns:
            List of match dictionaries with line info and context
        """
        await self.initialize()

        if not self.orchestrator.indexing_engine:
            return []

        tag = IndexTag(
            directory=str(Path(self.config.paths[0]).resolve()),
            branch="main",
            artifact_id="enhanced_full_text",
        )

        fulltext_index = self.orchestrator.indexing_engine.coordinator.get_index(
            "enhanced_full_text"
        )
        if fulltext_index and hasattr(fulltext_index, "search_in_file"):
            return await fulltext_index.search_in_file(file_path, query, tag, context_lines)
        return []

    async def get_similar_chunks(
        self,
        chunk_content: str,
        limit: int = 10,
        exclude_chunk_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find code chunks semantically similar to the given content.

        Args:
            chunk_content: Content to find similar chunks for
            limit: Maximum number of results
            exclude_chunk_id: Optional chunk ID to exclude from results

        Returns:
            List of similar chunk dictionaries
        """
        await self.initialize()

        if not self.orchestrator.indexing_engine:
            return []

        tag = IndexTag(
            directory=str(Path(self.config.paths[0]).resolve()),
            branch="main",
            artifact_id="enhanced_vectors",
        )

        vector_index = self.orchestrator.indexing_engine.coordinator.get_index(
            "enhanced_vectors"
        )
        if vector_index and hasattr(vector_index, "get_similar_chunks"):
            return await vector_index.get_similar_chunks(
                chunk_content, tag, limit, exclude_chunk_id
            )
        return []

    async def optimize_indexes(self) -> None:
        """Optimize all indexes for better performance."""
        await self.initialize()

        if not self.orchestrator.indexing_engine:
            return

        tag = IndexTag(
            directory=str(Path(self.config.paths[0]).resolve()),
            branch="main",
            artifact_id="*",
        )

        # Optimize vector index
        vector_index = self.orchestrator.indexing_engine.coordinator.get_index(
            "enhanced_vectors"
        )
        if vector_index and hasattr(vector_index, "optimize_index"):
            await vector_index.optimize_index(tag)

        logger.info("All indexes optimized")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.orchestrator.cleanup()


# Convenience functions for backward compatibility
async def index_search(
    query: str,
    config: SearchConfig,
    limit: int = 50,
) -> list[IndexSearchResult]:
    """Convenience function for enhanced search."""
    engine = IndexSearchEngine(config)
    try:
        return await engine.search(query, limit)
    finally:
        await engine.cleanup()


async def ensure_indexed(config: SearchConfig, force: bool = False) -> bool:
    """Convenience function to ensure codebase is indexed."""
    orchestrator = IndexingOrchestrator(config)
    try:
        await orchestrator.initialize()
        return await orchestrator.ensure_indexed(force_refresh=force)
    finally:
        await orchestrator.cleanup()
