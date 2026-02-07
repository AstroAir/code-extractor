"""
Data storage and persistence layer.

This module handles all data storage and persistence operations:
- Vector database integration and management (LanceDB, Qdrant, Chroma)
- Embedding providers (OpenAI, HuggingFace, local models)
- Qdrant client for vector similarity search
- Data serialization and deserialization
- Storage optimization and compression

The storage module provides a clean abstraction over different
storage backends, ensuring data persistence and efficient retrieval.
"""

from .qdrant_client import QdrantConfig, QdrantVectorStore, VectorSearchResult as QdrantSearchResult
from .vector_db import (
    ChromaProvider,
    EmbeddingConfig,
    EmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    LanceDBProvider,
    MultiProviderVectorManager,
    OpenAIEmbeddingProvider,
    QdrantProvider,
    VectorDatabase,
    VectorIndexManager,
    VectorSearchResult,
)

__all__ = [
    # Vector database abstractions
    "VectorDatabase",
    "LanceDBProvider",
    "QdrantProvider",
    "ChromaProvider",
    # Embedding providers
    "EmbeddingConfig",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    # Vector index management
    "VectorIndexManager",
    "MultiProviderVectorManager",
    "VectorSearchResult",
    # Qdrant integration
    "QdrantConfig",
    "QdrantVectorStore",
    "QdrantSearchResult",
]
