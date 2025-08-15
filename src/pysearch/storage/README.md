# Storage Module

The storage module provides data persistence and storage backend integration.

## Responsibilities

- **Vector Storage**: Vector database integration for semantic search
- **Data Persistence**: Efficient data serialization and storage
- **Backend Integration**: Support for multiple storage backends
- **Performance Optimization**: Storage-level performance optimizations

## Key Files

- `vector_db.py` - Enhanced vector database functionality (renamed from `enhanced_vector_db.py`)
- `qdrant_client.py` - Qdrant vector database client integration

## Storage Backends

1. **Qdrant**: Vector similarity search and storage
2. **Local Storage**: File-based storage for indexes and metadata
3. **Memory Cache**: In-memory storage for frequently accessed data

## Usage

```python
from pysearch.storage import QdrantVectorStore, QdrantConfig

config = QdrantConfig(host="localhost", port=6333)
vector_store = QdrantVectorStore(config)
```
