"""
Enhanced indexing features and advanced capabilities.

This module contains advanced indexing functionality that extends
the basic indexing system with sophisticated features:
- Enhanced indexing engine with advanced algorithms
- Seamless integration with the main search system
- Advanced code-aware chunking strategies
- Performance optimizations and monitoring

These features are designed to provide enterprise-grade search
capabilities while maintaining compatibility with the core system.
"""

from .base import EnhancedCodebaseIndex
from .coordinator import IndexCoordinator
from .engine import EnhancedIndexingEngine
from .locking import IndexLock

__all__ = [
    "EnhancedCodebaseIndex",
    "IndexCoordinator",
    "EnhancedIndexingEngine",
    "IndexLock",
]
