"""
Enhanced indexing integration module.

This module provides the main integration point for the enhanced code indexing
engine, allowing seamless integration with the existing pysearch system while
providing access to all enhanced features.

Classes:
    EnhancedSearchEngine: Main enhanced search engine
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
        >>> from pysearch.enhanced_integration import EnhancedSearchEngine
        >>> engine = EnhancedSearchEngine(config)
        >>> results = await engine.search("database connection", limit=10)

    Advanced search with filters:
        >>> results = await engine.enhanced_search(
        ...     query="user authentication",
        ...     languages=["python", "javascript"],
        ...     entity_types=["function", "class"],
        ...     semantic_threshold=0.8
        ... )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...core.config import SearchConfig
from ...analysis.content_addressing import IndexTag
from .engine import EnhancedIndexingEngine
from ...utils.logging_config import get_logger
from ...utils.performance_monitoring import PerformanceMonitor
from ...core.types import Language, OutputFormat

logger = get_logger()


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with additional metadata."""
    path: str
    content: str
    start_line: int
    end_line: int
    language: Language
    score: float

    # Enhanced metadata
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    similarity_score: Optional[float] = None
    complexity_score: Optional[float] = None
    quality_score: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    context: Optional[str] = None

    # Source information
    index_type: str = "unknown"
    chunk_id: Optional[str] = None


class SearchResultEnhancer:
    """Enhances search results with additional metadata and context."""

    def __init__(self, config: SearchConfig):
        self.config = config

    async def enhance_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        include_context: bool = True,
    ) -> List[EnhancedSearchResult]:
        """Enhance search results with additional metadata."""
        enhanced_results = []

        for result in results:
            enhanced_result = EnhancedSearchResult(
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
                enhanced_result.context = await self._extract_context(
                    enhanced_result, query
                )

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def _extract_context(
        self,
        result: EnhancedSearchResult,
        query: str,
    ) -> Optional[str]:
        """Extract relevant context around the search result."""
        try:
            # Read file and extract context
            file_path = Path(result.path)
            if not file_path.exists():
                return None

            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')

            # Get context lines around the match
            context_lines = 3
            start_idx = max(0, result.start_line - 1 - context_lines)
            end_idx = min(len(lines), result.end_line + context_lines)

            context_content = '\n'.join(lines[start_idx:end_idx])
            return context_content

        except Exception as e:
            logger.debug(f"Error extracting context for {result.path}: {e}")
            return None


class IndexingOrchestrator:
    """Orchestrates indexing operations across the enhanced system."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.indexing_engine: Optional[EnhancedIndexingEngine] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.last_index_time: Optional[float] = None

    async def initialize(self) -> None:
        """Initialize the indexing orchestrator."""
        if self.config.enable_enhanced_indexing:
            self.indexing_engine = EnhancedIndexingEngine(self.config)
            await self.indexing_engine.initialize()

            # Initialize performance monitoring
            cache_dir = self.config.resolve_cache_dir()
            self.performance_monitor = PerformanceMonitor(
                self.config, cache_dir)
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
                        logger.info(
                            f"Indexing progress: {progress.progress:.1%}")

                self.last_index_time = time.time()
                logger.info("Indexing completed successfully")
                return True

            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                return False

        return True

    async def _check_for_changes(self) -> bool:
        """Check if any files have changed since last indexing."""
        # This would implement change detection logic
        # For now, return False (no changes detected)
        return False

    async def get_health_status(self) -> Dict[str, Any]:
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


class EnhancedSearchEngine:
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
    ) -> List[EnhancedSearchResult]:
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
                artifact_id="*"
            )

            # Search code snippets
            try:
                snippets_index = self.orchestrator.indexing_engine.coordinator.get_index(
                    "enhanced_code_snippets"
                )
                if snippets_index:
                    snippet_results = await snippets_index.retrieve(query, tag, limit//2)
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
                    fulltext_results = await fulltext_index.retrieve(query, tag, limit//2)
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
                    vector_results = await vector_index.retrieve(query, tag, limit//2)
                    for result in vector_results:
                        result["index_type"] = "semantic"
                    all_results.extend(vector_results)
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")

        # Fallback to basic search if no enhanced results
        if not all_results:
            logger.info(
                "No enhanced search results, falling back to basic search")
            # TODO: Implement fallback to basic pysearch
            return []

        # Deduplicate and rank results
        deduplicated_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(deduplicated_results, query)

        # Limit results
        limited_results = ranked_results[:limit]

        # Enhance results with additional metadata
        enhanced_results = await self.result_enhancer.enhance_results(
            limited_results, query, include_context
        )

        return enhanced_results

    async def enhanced_search(
        self,
        query: str,
        languages: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None,
        semantic_threshold: float = 0.0,
        limit: int = 50,
    ) -> List[EnhancedSearchResult]:
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
            if (semantic_threshold > 0 and
                result.similarity_score is not None and
                    result.similarity_score < semantic_threshold):
                continue

            filtered_results.append(result)

        return filtered_results

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
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

    async def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        stats: dict[str, Any] = {
            "initialized": self._initialized,
            "enhanced_indexing_enabled": self.config.enable_enhanced_indexing,
        }

        if self.orchestrator.indexing_engine:
            index_stats = await self.orchestrator.indexing_engine.get_index_stats()
            stats.update(index_stats)

        health_status = await self.orchestrator.get_health_status()
        stats["health"] = health_status

        return stats

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.orchestrator.cleanup()


# Convenience functions for backward compatibility
async def enhanced_search(
    query: str,
    config: SearchConfig,
    limit: int = 50,
) -> List[EnhancedSearchResult]:
    """Convenience function for enhanced search."""
    engine = EnhancedSearchEngine(config)
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
