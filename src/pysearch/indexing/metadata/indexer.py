"""
Main metadata indexer implementation.

This module provides the MetadataIndexer class that coordinates the entire
metadata indexing process, including entity extraction, semantic analysis,
and database storage.

Classes:
    MetadataIndexer: Main metadata indexing coordinator

Features:
    - Comprehensive file and entity indexing
    - Semantic analysis and embedding generation
    - Incremental updates with change detection
    - Integration with entity extraction and analysis
    - Performance tracking and statistics
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ...analysis.graphrag.core import EntityExtractor
from ...analysis.language_detection import detect_language
from ...core.config import SearchConfig
from ...search.semantic_advanced import SemanticEmbedding
from ...utils.utils import read_text_safely
from ..indexer import Indexer
from .analysis import (
    calculate_entity_complexity,
    calculate_file_complexity,
    create_entity_text,
    extract_dependencies,
    extract_imports,
)
from .database import MetadataIndex
from .models import EntityMetadata, FileMetadata, IndexQuery, IndexStats

logger = logging.getLogger(__name__)


class MetadataIndexer:
    """
    Metadata indexer with comprehensive metadata support.

    This class extends the basic indexer with advanced metadata indexing,
    entity extraction, and semantic analysis capabilities.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.base_indexer = Indexer(config)
        self.entity_extractor = EntityExtractor()
        self.semantic_embedding = SemanticEmbedding()

        # Initialize metadata index
        cache_dir = config.resolve_cache_dir()
        self.metadata_index = MetadataIndex(cache_dir / "metadata.db")
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the metadata indexer."""
        await self.metadata_index.initialize()
        self._initialized = True
        logger.info("Metadata indexer initialized")

    async def build_index(
        self, include_semantic: bool = True, force_rebuild: bool = False
    ) -> IndexStats:
        """Build the comprehensive index."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        logger.info("Building enhanced index...")

        # Get file changes from base indexer
        changed_files, removed_files, total_files = self.base_indexer.scan()

        if force_rebuild:
            # Process all files - use scan to get all files
            all_changed, all_removed, total = self.base_indexer.scan()
            changed_files = all_changed

        logger.info(f"Processing {len(changed_files)} files for metadata indexing")

        # Prepare semantic embedding if needed
        if include_semantic and changed_files:
            documents = []
            for file_path in changed_files:
                content = read_text_safely(file_path)
                if content:
                    documents.append(content)

            if documents and not self.semantic_embedding.is_fitted:
                self.semantic_embedding.fit(documents)

        # Process each changed file
        for file_path in changed_files:
            await self._index_file(file_path, include_semantic)

        # Remove metadata for deleted files
        for file_path in removed_files:
            await self._remove_file_metadata(file_path)

        build_duration = time.time() - start_time
        stats = await self.metadata_index.get_stats()
        stats.build_duration = build_duration
        stats.last_build_time = time.time()

        logger.info(
            f"Enhanced index built in {build_duration:.2f}s: {stats.total_files} files, {stats.total_entities} entities"
        )
        return stats

    async def _index_file(self, file_path: Path, include_semantic: bool = True) -> None:
        """Index a single file with comprehensive metadata."""
        try:
            # Read file content
            content = read_text_safely(file_path)
            if not content:
                return

            # Get basic file info
            stat = file_path.stat()
            language = detect_language(file_path)
            lines = content.split("\n")
            line_count = len(lines)

            # Calculate complexity score (simple metric)
            complexity_score = calculate_file_complexity(content, language)

            # Extract entities
            entities = await self.entity_extractor.extract_from_file(file_path)

            # Generate semantic summary if requested
            semantic_summary = None
            if include_semantic and self.semantic_embedding.is_fitted:
                # Create a summary from first few lines and docstrings
                summary_parts = []
                if len(lines) > 0:
                    summary_parts.extend(lines[:5])  # First 5 lines

                for entity in entities:
                    if entity.docstring:
                        summary_parts.append(entity.docstring)

                if summary_parts:
                    summary_text = " ".join(summary_parts)[:500]  # Limit length
                    semantic_summary = summary_text

            # Extract imports and dependencies (simplified)
            imports = extract_imports(content, language)
            dependencies = extract_dependencies(content, language)

            # Create file metadata
            file_metadata = FileMetadata(
                file_path=str(file_path),
                size=stat.st_size,
                mtime=stat.st_mtime,
                language=language.value if hasattr(language, "value") else str(language),
                line_count=line_count,
                entity_count=len(entities),
                complexity_score=complexity_score,
                semantic_summary=semantic_summary,
                imports=imports,
                dependencies=dependencies,
                last_indexed=time.time(),
            )

            # Store file metadata
            await self.metadata_index.add_file_metadata(file_metadata)

            # Process entities
            for entity in entities:
                entity_complexity = calculate_entity_complexity(entity, content)

                # Generate semantic embedding if requested
                semantic_embedding = None
                if include_semantic and self.semantic_embedding.is_fitted:
                    entity_text = create_entity_text(entity)
                    embedding_vector = self.semantic_embedding.transform(entity_text)
                    if embedding_vector:
                        # Convert sparse to dense
                        max_dim = max(embedding_vector.keys()) + 1 if embedding_vector else 0
                        dense_vector = [0.0] * max_dim
                        for dim, value in embedding_vector.items():
                            dense_vector[dim] = value
                        semantic_embedding = dense_vector

                # Create entity metadata
                entity_metadata = EntityMetadata(
                    entity_id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    file_path=str(file_path),
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    signature=entity.signature,
                    docstring=entity.docstring,
                    language=(
                        entity.language.value
                        if hasattr(entity.language, "value")
                        else str(entity.language)
                    ),
                    scope=entity.scope,
                    complexity_score=entity_complexity,
                    semantic_embedding=semantic_embedding,
                    properties=entity.properties,
                )

                # Store entity metadata
                await self.metadata_index.add_entity_metadata(entity_metadata)

            logger.debug(f"Indexed {file_path}: {len(entities)} entities")

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")

    async def _remove_file_metadata(self, file_path: Path) -> None:
        """Remove metadata for a deleted file."""
        try:
            await self.metadata_index.delete_file_metadata(str(file_path))
            logger.debug(f"File removed from index: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove metadata for {file_path}: {e}")

    async def query_index(self, query: IndexQuery) -> dict[str, Any]:
        """Query the enhanced index."""
        if not self._initialized:
            await self.initialize()

        results: dict[str, Any] = {"files": [], "entities": [], "stats": {}}

        # Query files
        files = await self.metadata_index.query_files(query)
        results["files"] = [
            {
                "path": f.file_path,
                "language": f.language,
                "size": f.size,
                "line_count": f.line_count,
                "entity_count": f.entity_count,
                "complexity": f.complexity_score,
            }
            for f in files
        ]

        # Query entities if requested
        if query.include_entities:
            entities = await self.metadata_index.query_entities(query)
            results["entities"] = [
                {
                    "id": e.entity_id,
                    "name": e.name,
                    "type": e.entity_type,
                    "file": e.file_path,
                    "line": e.start_line,
                    "signature": e.signature,
                    "complexity": e.complexity_score,
                }
                for e in entities
            ]

        # Get stats
        stats = await self.metadata_index.get_stats()
        results["stats"] = {
            "total_files": stats.total_files,
            "total_entities": stats.total_entities,
            "languages": stats.languages,
            "entity_types": stats.entity_types,
        }

        return results

    async def close(self) -> None:
        """Close the metadata indexer."""
        await self.metadata_index.close()
        logger.info("Metadata indexer closed")
