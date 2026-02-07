"""
Enhanced index implementations for the pysearch indexing engine.

This package contains concrete implementations of the CodebaseIndex
interface, providing different types of indexes for comprehensive code search
and analysis capabilities.

Index Types:
    - CodeSnippetsIndex: Tree-sitter based code structure indexing
    - FullTextIndex: SQLite FTS5 based full-text search
    - ChunkIndex: Code-aware chunking for embeddings
    - VectorIndex: Vector database integration for semantic search
    - DependencyIndex: Code dependency and relationship tracking

Each index type implements the CodebaseIndex interface and can be
used independently or in combination through the IndexCoordinator.
"""

from .chunk_index import ChunkIndex
from .code_snippets_index import CodeSnippetsIndex
from .full_text_index import FullTextIndex

__all__ = [
    "CodeSnippetsIndex",
    "FullTextIndex",
    "ChunkIndex",
]

# Vector index is optional due to dependencies
try:
    from .vector_index import VectorIndex  # noqa: F401

    __all__.append("VectorIndex")
except ImportError:
    pass

# Dependency index is optional - not implemented yet
# try:
#     from .dependency_index import EnhancedDependencyIndex
#     __all__.append("EnhancedDependencyIndex")
# except ImportError:
#     pass
