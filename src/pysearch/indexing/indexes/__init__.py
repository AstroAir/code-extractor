"""
Enhanced index implementations for the pysearch indexing engine.

This package contains concrete implementations of the EnhancedCodebaseIndex
interface, providing different types of indexes for comprehensive code search
and analysis capabilities.

Index Types:
    - CodeSnippetsIndex: Tree-sitter based code structure indexing
    - FullTextIndex: SQLite FTS5 based full-text search
    - ChunkIndex: Code-aware chunking for embeddings
    - VectorIndex: Vector database integration for semantic search
    - DependencyIndex: Code dependency and relationship tracking

Each index type implements the EnhancedCodebaseIndex interface and can be
used independently or in combination through the IndexCoordinator.
"""

from .code_snippets_index import EnhancedCodeSnippetsIndex
from .full_text_index import EnhancedFullTextIndex
from .chunk_index import EnhancedChunkIndex

__all__ = [
    "EnhancedCodeSnippetsIndex",
    "EnhancedFullTextIndex", 
    "EnhancedChunkIndex",
]

# Vector index is optional due to dependencies
try:
    from .vector_index import EnhancedVectorIndex
    __all__.append("EnhancedVectorIndex")
except ImportError:
    pass

# Dependency index is optional
try:
    from .dependency_index import EnhancedDependencyIndex
    __all__.append("EnhancedDependencyIndex")
except ImportError:
    pass
