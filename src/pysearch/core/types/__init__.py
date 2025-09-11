"""
Core type definitions for pysearch.

This package contains all the fundamental data types, enumerations, and data classes
used throughout the pysearch system. The types are organized into logical modules:

- basic_types: Core search types, enums, and data structures
- graphrag_types: GraphRAG-specific types for knowledge graphs

For backward compatibility, all types are re-exported from this package.
"""

# Import all types from submodules for backward compatibility
from .basic_types import (
    ASTFilters,
    FileMetadata,
    Language,
    MatchSpan,
    MetadataFilters,
    OutputFormat,
    Query,
    SearchItem,
    SearchResult,
    SearchStats,
)
from .graphrag_types import (
    CodeEntity,
    EntityRelationship,
    EntityType,
    GraphRAGQuery,
    GraphRAGResult,
    KnowledgeGraph,
    RelationType,
)

# Re-export everything for backward compatibility
__all__ = [
    # Basic types
    "ASTFilters",
    "FileMetadata", 
    "Language",
    "MatchSpan",
    "MetadataFilters",
    "OutputFormat",
    "Query",
    "SearchItem",
    "SearchResult",
    "SearchStats",
    # GraphRAG types
    "CodeEntity",
    "EntityRelationship",
    "EntityType",
    "GraphRAGQuery",
    "GraphRAGResult",
    "KnowledgeGraph",
    "RelationType",
]
