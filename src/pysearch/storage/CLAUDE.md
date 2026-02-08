# Storage Module

[根目录](../../../CLAUDE.md) > **storage**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Added Qdrant client documentation
- Enhanced vector database abstraction documentation
- Synchronized with current project structure

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive storage module documentation

---

## Module Responsibility

The **Storage** module provides vector database integration and storage backends for semantic search and GraphRAG features.

### Key Responsibilities
1. **Vector Database**: Abstraction layer for vector storage
2. **Qdrant Client**: Qdrant vector database integration
3. **Vector Operations**: Insert, search, and delete vectors

---

## Entry and Startup

### Main Entry Points
- **`vector_db.py`** - Vector database abstraction
  - `VectorStore` - Vector storage interface
  - `QdrantVectorStore` - Qdrant implementation
  - `FaissVectorStore` - FAISS implementation

- **`qdrant_client.py`** - Qdrant client
  - `QdrantClient` - Qdrant database client
  - `QdrantConfig` - Qdrant configuration

---

## Public API

### Vector Store

```python
from pysearch.storage import VectorStore

# Initialize vector store
store = VectorStore(
    backend="qdrant",
    host="localhost",
    port=6333,
    collection_name="pysearch_vectors"
)

# Insert vectors
store.insert(
    vectors=[embedding1, embedding2],
    payloads=[{"id": 1}, {"id": 2}],
    ids=["doc1", "doc2"]
)

# Search similar vectors
results = store.search(
    query_vector=embedding,
    limit=10,
    score_threshold=0.7
)
```

### Qdrant Client

```python
from pysearch.storage.qdrant_client import QdrantClient, QdrantConfig

config = QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="pysearch_vectors",
    vector_size=384,
    distance_metric="Cosine"
)

client = QdrantClient(config)

# Create collection
client.create_collection()

# Upsert points
client.upsert(points=[...])

# Search
results = client.search(query_vector, limit=10)
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.core.config` - Configuration

### External Dependencies
- `qdrant-client>=1.7.0` - Qdrant vector database client
- `numpy>=1.24.0` - Numerical operations

### Optional Dependencies
- `[vector]` extra includes:
  - `faiss-cpu>=1.7.0` - FAISS vector store

---

## Data Models

### Qdrant Types
- `QdrantConfig` - Qdrant configuration
- `QdrantPoint` - Qdrant point structure
- `QdrantSearchResult` - Search result

### Vector Store Types
- `VectorStore` - Vector store interface
- `VectorSearchResult` - Search result structure
- `VectorConfig` - Vector store configuration

---

## Testing

### Test Directory
- `tests/unit/storage/` - Storage module tests
  - `test_vector_db.py` - Vector store tests
  - `test_qdrant_client.py` - Qdrant client tests
- `tests/integration/test_qdrant_integration.py` - Integration tests

### Running Tests
```bash
pytest tests/unit/storage/ -v
pytest tests/integration/test_qdrant_integration.py -v
```

---

## Common Issues and Solutions

### Issue 1: Qdrant connection failed
**Symptoms**: Connection refused error
**Solution**: Ensure Qdrant is running:
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Issue 2: Vector dimension mismatch
**Symptoms**: Dimension validation error
**Solution**: Ensure vector size matches configuration:
```python
config = SearchConfig(
    qdrant_vector_size=384,  # Match your embedding model
    embedding_model="all-MiniLM-L6-v2"
)
```

### Issue 3: Search slow on large collections
**Symptoms**: Slow search response
**Solution**: Use indexing and limit results:
```python
results = client.search(
    query_vector,
    limit=100,  # Limit results
    search_params={"hnsw_ef": 128}  # Tune HNSW
)
```

---

## Related Files

### Storage Module Files
- `src/pysearch/storage/__init__.py`
- `src/pysearch/storage/vector_db.py` - Vector database abstraction
- `src/pysearch/storage/qdrant_client.py` - Qdrant client

---

## Module Structure

```
storage/
├── __init__.py
├── vector_db.py      # Vector database abstraction
└── qdrant_client.py  # Qdrant client implementation
```
