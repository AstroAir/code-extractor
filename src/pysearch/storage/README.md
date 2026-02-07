# Storage Module

The storage module provides vector database integration, embedding generation, and data persistence for semantic code search.

## Responsibilities

- **Vector Storage**: Unified `VectorDatabase` ABC with LanceDB, Qdrant, and Chroma providers
- **Embedding Providers**: OpenAI and HuggingFace (sentence-transformers) embedding backends
- **Qdrant Integration**: Standalone `QdrantVectorStore` with `AsyncQdrantClient`, retry logic, batch search
- **Index Management**: `VectorIndexManager` for coordinating chunking, embedding, storage, and retrieval

## Key Files

- `vector_db.py` - Vector database abstraction layer: `VectorDatabase` ABC, `LanceDBProvider`, `QdrantProvider`, `ChromaProvider`, `EmbeddingProvider`, `VectorIndexManager`, `MultiProviderVectorManager`
- `qdrant_client.py` - Qdrant client: `QdrantVectorStore` with async context manager, batch search, scalar quantization
- `__init__.py` - Module exports for all public classes
- `CLAUDE.md` - Detailed module documentation with class signatures and usage examples

## Storage Backends

1. **LanceDB** (default): Serverless vector database, no external service required
2. **Qdrant**: High-performance vector similarity search with scalar quantization support
3. **Chroma**: Open-source embedding database with HNSW index

## Embedding Providers

1. **OpenAI**: `text-embedding-ada-002` and compatible models via async API
2. **HuggingFace**: `sentence-transformers` models (e.g., `all-MiniLM-L6-v2`) via local inference

## Usage

```python
# Standalone Qdrant (with async context manager)
from pysearch.storage import QdrantVectorStore, QdrantConfig

async with QdrantVectorStore(QdrantConfig(host="localhost", port=6333)) as store:
    ids = await store.add_vectors("collection", vectors, metadata)
    results = await store.search_similar(query_vector, top_k=10)

# Unified VectorIndexManager (provider-agnostic)
from pysearch.storage import VectorIndexManager, EmbeddingConfig
from pathlib import Path

config = EmbeddingConfig(provider="huggingface", model_name="all-MiniLM-L6-v2")
manager = VectorIndexManager(Path("./db"), config, provider="lancedb")  # or "qdrant", "chroma"
```
