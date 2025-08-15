"""
Indexing and caching systems for efficient file processing.

This module handles all aspects of file indexing and caching:
- Basic file indexing with metadata tracking
- Metadata indexing for advanced queries
- Cache management for performance optimization
- Enhanced indexing features (in enhanced/ subdirectory)
- Specialized index implementations

The indexing module ensures fast and efficient access to file content
by maintaining up-to-date indexes and intelligent caching strategies.
"""

from .cache_manager import CacheManager
from .indexer import Indexer
from .metadata import MetadataIndexer, IndexQuery

__all__ = [
    # Core indexing
    "Indexer",
    "CacheManager",
    # Metadata indexing
    "MetadataIndexer",
    "IndexQuery",
]
