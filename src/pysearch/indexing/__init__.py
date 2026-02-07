"""
Indexing and caching systems for efficient file processing.

This module handles all aspects of file indexing and caching:
- Basic file indexing with metadata tracking
- Metadata indexing for advanced queries
- Cache management for performance optimization
- Advanced indexing features (in advanced/ subdirectory)
- Specialized index implementations (in indexes/ subdirectory)
- Cache backends and statistics (in cache/ subdirectory)

The indexing module ensures fast and efficient access to file content
by maintaining up-to-date indexes and intelligent caching strategies.
"""

from .cache import CacheManager
from .indexer import Indexer
from .metadata import IndexQuery, MetadataIndexer

__all__ = [
    # Core indexing
    "Indexer",
    "CacheManager",
    # Metadata indexing
    "MetadataIndexer",
    "IndexQuery",
]

# Advanced indexing classes (lazy import to avoid circular imports)
def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy import for advanced indexing classes."""
    _advanced_exports = {
        "CodebaseIndex": ".advanced.base",
        "IndexCoordinator": ".advanced.coordinator",
        "IndexingEngine": ".advanced.engine",
        "IndexLock": ".advanced.locking",
        "ChunkingEngine": ".advanced.chunking",
        "IndexSearchEngine": ".advanced.integration",
    }
    if name in _advanced_exports:
        import importlib

        module = importlib.import_module(_advanced_exports[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
