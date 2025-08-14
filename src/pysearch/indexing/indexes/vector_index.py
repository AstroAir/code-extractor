"""
Enhanced vector index implementation for semantic code search.

This module implements a comprehensive vector index that combines code chunking
with vector embeddings for semantic similarity search, extending beyond
Continue's basic vector indexing with multiple providers and advanced features.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..advanced.chunking import ChunkingEngine, ChunkingConfig, ChunkingStrategy
from ...analysis.content_addressing import IndexTag, IndexingProgressUpdate, MarkCompleteCallback, PathAndCacheKey, RefreshIndexResults
from ..advanced.engine import EnhancedCodebaseIndex
from ...storage.vector_db import EmbeddingConfig, VectorIndexManager
from ...analysis.language_detection import detect_language
from ...utils.logging_config import get_logger
from ...utils.utils import read_text_safely

logger = get_logger()


class EnhancedVectorIndex(EnhancedCodebaseIndex):
    """
    Enhanced vector index with semantic search capabilities.

    This index chunks code intelligently and creates vector embeddings
    for semantic similarity search, supporting multiple vector databases
    and embedding providers.
    """

    artifact_id = "enhanced_vectors"
    relative_expected_time = 3.0  # Higher cost due to embedding generation

    def __init__(self, config: Any) -> None:
        self.config = config
        self.cache_dir = config.resolve_cache_dir()

        # Initialize chunking engine
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=getattr(config, 'chunk_size', 1000),
            min_chunk_size=50,
            overlap_size=100,
            respect_boundaries=True,
        )
        self.chunking_engine = ChunkingEngine(chunking_config)

        # Initialize vector database
        embedding_config = EmbeddingConfig(
            provider=getattr(config, 'embedding_provider', 'openai'),
            model_name=getattr(config, 'embedding_model',
                               'text-embedding-ada-002'),
            batch_size=getattr(config, 'embedding_batch_size', 100),
            api_key=getattr(config, 'openai_api_key', None),
        )

        self.vector_manager = VectorIndexManager(
            self.cache_dir / "vectors",
            embedding_config,
            provider=getattr(config, 'vector_db_provider', 'lancedb')
        )

    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: Optional[str] = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """Update the vector index."""
        collection_name = self.vector_manager.get_collection_name(tag)

        # Process compute operations (new files)
        total_operations = len(results.compute) + len(results.delete) + \
            len(results.add_tag) + len(results.remove_tag)
        completed_operations = 0

        for item in results.compute:
            yield IndexingProgressUpdate(
                progress=completed_operations / max(total_operations, 1),
                description=f"Generating embeddings for {Path(item.path).name}",
                status="indexing"
            )

            try:
                # Read and chunk file
                content = read_text_safely(Path(item.path))
                if not content:
                    continue
                chunks = await self.chunking_engine.chunk_file(item.path, content)

                if chunks:
                    # Index chunks in vector database
                    await self.vector_manager.index_chunks(chunks, collection_name)

                mark_complete([item], "compute")
                completed_operations += 1

            except Exception as e:
                logger.error(f"Error processing file {item.path}: {e}")
                completed_operations += 1

        # Process add_tag operations
        for item in results.add_tag:
            yield IndexingProgressUpdate(
                progress=completed_operations / max(total_operations, 1),
                description=f"Adding tag for {Path(item.path).name}",
                status="indexing"
            )

            try:
                # For add_tag, the content should already exist in global cache
                # We just need to associate it with this collection
                # This is a simplified implementation - in practice, we'd
                # retrieve from global cache and add to this collection

                mark_complete([item], "add_tag")
                completed_operations += 1

            except Exception as e:
                logger.error(f"Error adding tag for {item.path}: {e}")
                completed_operations += 1

        # Process remove_tag operations
        for item in results.remove_tag:
            yield IndexingProgressUpdate(
                progress=completed_operations / max(total_operations, 1),
                description=f"Removing tag for {Path(item.path).name}",
                status="indexing"
            )

            try:
                # Remove vectors for this file from the collection
                # This would require tracking chunk IDs by file path
                chunk_ids = [f"{item.path}:{item.cache_key}"]  # Simplified
                await self.vector_manager.delete_chunks(chunk_ids, collection_name)

                mark_complete([item], "remove_tag")
                completed_operations += 1

            except Exception as e:
                logger.error(f"Error removing tag for {item.path}: {e}")
                completed_operations += 1

        # Process delete operations
        for item in results.delete:
            yield IndexingProgressUpdate(
                progress=completed_operations / max(total_operations, 1),
                description=f"Deleting vectors for {Path(item.path).name}",
                status="indexing"
            )

            try:
                # Delete all vectors for this file
                chunk_ids = [f"{item.path}:{item.cache_key}"]  # Simplified
                await self.vector_manager.delete_chunks(chunk_ids, collection_name)

                mark_complete([item], "delete")
                completed_operations += 1

            except Exception as e:
                logger.error(f"Error deleting vectors for {item.path}: {e}")
                completed_operations += 1

        yield IndexingProgressUpdate(
            progress=1.0,
            description="Vector indexing complete",
            status="done"
        )

    async def retrieve(
        self,
        query: str,
        tag: IndexTag,
        limit: int = 50,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically similar code chunks."""
        collection_name = self.vector_manager.get_collection_name(tag)

        # Build filters from kwargs
        filters = {}
        if "language" in kwargs:
            filters["language"] = kwargs["language"]
        if "file_path" in kwargs:
            filters["file_path"] = kwargs["file_path"]
        if "chunk_type" in kwargs:
            filters["chunk_type"] = kwargs["chunk_type"]

        # Get similarity threshold
        similarity_threshold = kwargs.get("similarity_threshold", 0.0)

        try:
            # Perform vector search
            results = await self.vector_manager.search(
                query=query,
                collection_name=collection_name,
                limit=limit,
                filters=filters,
                similarity_threshold=similarity_threshold,
            )

            # Convert to standard format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "similarity_score": result.similarity_score,
                    "language": result.metadata.get("language"),
                    "chunk_type": result.metadata.get("chunk_type"),
                    "entity_name": result.metadata.get("entity_name"),
                    "complexity_score": result.metadata.get("complexity_score", 0.0),
                    "quality_score": result.metadata.get("quality_score", 0.0),
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving from vector index: {e}")
            return []

    async def get_similar_chunks(
        self,
        chunk_content: str,
        tag: IndexTag,
        limit: int = 10,
        exclude_chunk_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find chunks similar to the given content."""
        collection_name = self.vector_manager.get_collection_name(tag)

        try:
            # Generate embedding for the chunk content
            query_vector = await self.vector_manager.embedding_provider.embed_query(chunk_content)

            # Search for similar vectors
            results = await self.vector_manager.vector_db.search_vectors(
                collection_name, query_vector, limit + 1  # +1 to account for self-match
            )

            # Filter out the chunk itself if specified
            filtered_results = []
            for result in results:
                if exclude_chunk_id and result.chunk_id == exclude_chunk_id:
                    continue
                filtered_results.append({
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "file_path": result.file_path,
                    "similarity_score": result.similarity_score,
                })

                if len(filtered_results) >= limit:
                    break

            return filtered_results

        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []

    async def get_statistics(self, tag: IndexTag) -> Dict[str, Any]:
        """Get statistics for this vector index."""
        collection_name = self.vector_manager.get_collection_name(tag)

        try:
            stats = await self.vector_manager.get_collection_stats(collection_name)
            return {
                "total_vectors": stats.get("total_vectors", 0),
                "provider": stats.get("provider", "unknown"),
                "dimensions": stats.get("dimensions", 0),
                "embedding_provider": self.vector_manager.embedding_config.provider,
                "embedding_model": self.vector_manager.embedding_config.model_name,
            }
        except Exception as e:
            logger.error(f"Error getting vector index statistics: {e}")
            return {}

    async def optimize_index(self, tag: IndexTag) -> None:
        """Optimize the vector index for better performance."""
        collection_name = self.vector_manager.get_collection_name(tag)

        try:
            await self.vector_manager.optimize_collection(collection_name)
            logger.info(f"Optimized vector index for tag: {tag.to_string()}")
        except Exception as e:
            logger.error(f"Error optimizing vector index: {e}")

    async def rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        boost_factors: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using additional signals.

        Args:
            results: Initial search results
            query: Original search query
            boost_factors: Optional boost factors for different attributes

        Returns:
            Re-ranked results
        """
        if not results:
            return results

        boost_factors = boost_factors or {
            "quality_score": 0.2,
            "complexity_score": 0.1,
            "exact_match": 0.3,
            "entity_name_match": 0.2,
        }

        # Calculate enhanced scores
        for result in results:
            base_score = result.get("similarity_score", 0.0)
            enhanced_score = base_score

            # Quality boost
            quality_score = result.get("quality_score", 0.0)
            enhanced_score += quality_score * \
                boost_factors.get("quality_score", 0.0)

            # Complexity boost (moderate complexity preferred)
            complexity_score = result.get("complexity_score", 0.0)
            complexity_boost = 1.0 - \
                abs(complexity_score - 0.5) * 2  # Peak at 0.5
            enhanced_score += complexity_boost * \
                boost_factors.get("complexity_score", 0.0)

            # Exact match boost
            if query.lower() in result.get("content", "").lower():
                enhanced_score += boost_factors.get("exact_match", 0.0)

            # Entity name match boost
            entity_name = result.get("entity_name", "")
            if entity_name and query.lower() in entity_name.lower():
                enhanced_score += boost_factors.get("entity_name_match", 0.0)

            result["enhanced_score"] = enhanced_score

        # Sort by enhanced score
        results.sort(key=lambda x: x.get("enhanced_score", 0.0), reverse=True)

        return results
