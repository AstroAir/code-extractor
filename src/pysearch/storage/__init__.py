"""
Data storage and persistence layer.

This module handles all data storage and persistence operations:
- Vector database integration and management
- Qdrant client for vector similarity search
- Data serialization and deserialization
- Storage optimization and compression

The storage module provides a clean abstraction over different
storage backends, ensuring data persistence and efficient retrieval.
"""

from .qdrant_client import QdrantConfig, QdrantVectorStore

__all__ = [
    # Qdrant integration
    "QdrantConfig",
    "QdrantVectorStore",
]
