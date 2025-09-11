"""
Metadata indexing package for pysearch.

This package provides comprehensive metadata indexing capabilities with
entity-level indexing, semantic metadata, and efficient querying.

Public API:
    EntityMetadata: Metadata for a code entity in the index
    FileMetadata: Enhanced metadata for a file in the index
    IndexQuery: Query specification for the enhanced index
    IndexStats: Statistics for the enhanced index
    MetadataIndex: SQLite-based metadata storage and querying
    MetadataIndexer: Main metadata indexing coordinator

The package is organized into focused modules:
    - models: Data structures and models
    - database: Database operations and storage
    - indexer: Main indexing coordinator
    - analysis: Analysis utilities and helpers
"""

from .database import MetadataIndex
from .indexer import MetadataIndexer
from .models import EntityMetadata, FileMetadata, IndexQuery, IndexStats

__all__ = [
    # Data models
    "EntityMetadata",
    "FileMetadata", 
    "IndexQuery",
    "IndexStats",
    # Core classes
    "MetadataIndex",
    "MetadataIndexer",
]
