"""
Metadata indexing system for pysearch with comprehensive metadata support.

This module provides backward compatibility for the refactored metadata indexing
system. The implementation has been split into focused modules for better
maintainability while preserving the public API.

Classes:
    MetadataIndex: Comprehensive metadata index for files and entities
    MetadataIndexer: Extended indexer with metadata and entity support
    IndexQuery: Query interface for the metadata index
    IndexStats: Statistics and analytics for the index

Features:
    - File-level and entity-level metadata indexing
    - Semantic metadata extraction and storage
    - Incremental updates with change detection
    - Efficient querying with multiple filter criteria
    - Integration with GraphRAG knowledge graphs
    - Performance analytics and optimization
    - Persistent storage with compression

Example:
    Basic metadata indexing:
        >>> from pysearch.indexing.metadata import MetadataIndexer
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["./src"])
        >>> indexer = MetadataIndexer(config)
        >>> await indexer.build_index()
        >>>
        >>> # Query the index
        >>> from pysearch.indexing.metadata import IndexQuery
        >>> query = IndexQuery(
        ...     entity_types=["function", "class"],
        ...     languages=["python"],
        ...     min_lines=10
        ... )
        >>> results = await indexer.query_index(query)

    Advanced metadata indexing:
        >>> # Index with semantic metadata
        >>> await indexer.build_index(include_semantic=True)
        >>>
        >>> # Query with semantic filters
        >>> query = IndexQuery(
        ...     semantic_query="database operations",
        ...     similarity_threshold=0.7
        ... )
        >>> results = await indexer.query_index(query)
"""

from __future__ import annotations

# Import everything from the new modular structure
from .metadata.models import EntityMetadata, FileMetadata, IndexQuery, IndexStats
from .metadata.database import MetadataIndex
from .metadata.indexer import MetadataIndexer
