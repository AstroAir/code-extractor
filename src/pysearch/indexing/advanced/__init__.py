"""
Indexing features and capabilities.

This module contains indexing functionality that extends
the basic indexing system with sophisticated features:
- Index search engine with multiple algorithms
- Seamless integration with the main search system
- Code-aware chunking strategies
- Performance optimizations and monitoring

These features are designed to provide enterprise-grade search
capabilities while maintaining compatibility with the core system.
"""

from .base import CodebaseIndex
from .chunking import ChunkingEngine
from .coordinator import IndexCoordinator
from .engine import IndexingEngine
from .integration import IndexSearchEngine
from .locking import IndexLock

__all__ = [
    "CodebaseIndex",
    "ChunkingEngine",
    "IndexCoordinator",
    "IndexingEngine",
    "IndexSearchEngine",
    "IndexLock",
]
