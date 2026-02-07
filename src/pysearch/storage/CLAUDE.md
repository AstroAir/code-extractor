# Storage Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **storage**

---

## Change Log (Changelog)

### 2026-02-07 - Comprehensive Audit & Fixes
- Added QdrantProvider and ChromaProvider to vector_db.py (unified VectorDatabase ABC)
- Added HuggingFaceEmbeddingProvider for sentence-transformers support
- Fixed SQL injection vulnerability in LanceDB filter/delete operations
- Switched QdrantVectorStore to use AsyncQdrantClient with sync fallback
- Added async context manager support (__aenter__/__aexit__)
- Implemented enable_compression via scalar quantization
- Added batch_search, list_collections, delete_collection to QdrantVectorStore
- Implemented cleanup_orphaned_vectors with real logic for all providers
- Fixed lambda closure in batch upsert loop
- Updated __init__.py exports to include all public classes
- Fixed test @patch paths from pysearch.qdrant_client → pysearch.storage.qdrant_client
- Aligned documentation with actual code

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive Storage module documentation

---

## Module Responsibility

The **Storage** module provides data persistence and storage backend integration:

1. **Vector Storage**: Vector database integration for semantic search (LanceDB, Qdrant, Chroma)
2. **Embedding Providers**: Multiple embedding backends (OpenAI, HuggingFace/sentence-transformers)
3. **Unified Abstraction**: VectorDatabase ABC with interchangeable providers
4. **Performance Optimization**: Batch operations, async I/O, scalar quantization

---

## Key Files

| File | Purpose | Description |
|------|---------|-------------|
| `vector_db.py` | Vector Database Abstraction | VectorDatabase ABC, LanceDBProvider, QdrantProvider, ChromaProvider, EmbeddingProviders, VectorIndexManager |
| `qdrant_client.py` | Qdrant Client | Standalone QdrantVectorStore with AsyncQdrantClient, retry logic, batch search |
| `__init__.py` | Module Init | Module initialization and all public exports |
| `README.md` | Module Docs | Storage module overview |

---

## Vector Database (vector_db.py)

### Overview
Provides a unified vector database abstraction layer with multiple providers and embedding backends.

### Key Classes

```python
@dataclass
class VectorSearchResult:
    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    similarity_score: float
    metadata: dict[str, Any]

@dataclass
class EmbeddingConfig:
    provider: str = "openai"       # "openai", "huggingface"
    model_name: str = "text-embedding-ada-002"
    batch_size: int = 100
    max_tokens: int = 8192
    dimensions: int = 1536
    api_key: str | None = None
    base_url: str | None = None

class EmbeddingProvider(ABC):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]
    async def embed_query(self, query: str) -> list[float]
    def dimensions(self) -> int  # property

class OpenAIEmbeddingProvider(EmbeddingProvider): ...
class HuggingFaceEmbeddingProvider(EmbeddingProvider): ...

class VectorDatabase(ABC):
    async def create_collection(self, collection_name: str) -> None
    async def insert_vectors(self, collection_name, vectors: list[tuple]) -> None
    async def search_vectors(self, collection_name, query_vector, limit, filters) -> list[VectorSearchResult]
    async def delete_vectors(self, collection_name, vector_ids: list[str]) -> None
    async def collection_exists(self, collection_name: str) -> bool

class LanceDBProvider(VectorDatabase): ...
class QdrantProvider(VectorDatabase): ...   # bridges to QdrantVectorStore
class ChromaProvider(VectorDatabase): ...

class VectorIndexManager:
    def __init__(self, db_path, embedding_config, provider="lancedb")
    async def index_chunks(self, chunks, collection_name) -> None
    async def search(self, query, collection_name, limit, filters, similarity_threshold) -> list[VectorSearchResult]
    async def delete_chunks(self, chunk_ids, collection_name) -> None
    async def update_chunks(self, chunks, collection_name) -> None
    def get_collection_name(self, tag: IndexTag) -> str
    async def get_collection_stats(self, collection_name) -> dict
    async def optimize_collection(self, collection_name) -> None
    async def cleanup_orphaned_vectors(self, collection_name, valid_file_paths=None) -> int

class MultiProviderVectorManager:
    async def add_provider(self, name, provider_type) -> None
    async def index_chunks_all_providers(self, chunks, collection_name) -> None
    async def search_best_provider(self, query, collection_name, limit, filters) -> list[VectorSearchResult]
    async def get_all_stats(self) -> dict
```

### Supported Backends
- **LanceDB**: Serverless vector database (default)
- **Qdrant**: High-performance vector similarity search with scalar quantization
- **Chroma**: Open-source embedding database with HNSW index

---

## Qdrant Client (qdrant_client.py)

### Overview
Standalone Qdrant client with AsyncQdrantClient support, retry logic, batch search, and async context manager.

### Key Classes
```python
@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    api_key: str | None = None
    https: bool = False
    timeout: float = 30.0
    collection_name: str = "pysearch_vectors"
    vector_size: int = 384
    distance_metric: str = "Cosine"  # Cosine, Dot, Euclid
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    enable_compression: bool = False  # enables scalar INT8 quantization

@dataclass
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, Any]
    vector: list[float] | None = None

class QdrantVectorStore:
    def __init__(self, config: QdrantConfig)
    async def initialize(self) -> None
    async def __aenter__(self) -> QdrantVectorStore  # context manager
    async def __aexit__(...) -> None
    async def create_collection(self, name, vector_size, distance_metric) -> None
    async def add_vectors(self, collection_name, vectors, metadata, ids) -> list[str]
    async def search_similar(self, query_vector, collection_name, top_k, filter_conditions, score_threshold) -> list[VectorSearchResult]
    async def batch_search(self, query_vectors, collection_name, top_k, filter_conditions, score_threshold) -> list[list[VectorSearchResult]]
    async def delete_vectors(self, collection_name, vector_ids) -> None
    async def update_vector_metadata(self, collection_name, vector_id, metadata) -> None
    async def get_collection_info(self, collection_name) -> dict
    async def list_collections(self) -> list[str]
    async def delete_collection(self, collection_name) -> None
    async def close(self) -> None
    def is_available(self) -> bool
```

### Utility Functions
```python
def normalize_vector(vector: list[float]) -> list[float]
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float
```

---

## Usage Examples

### Basic Qdrant Vector Storage
```python
from pysearch.storage import QdrantVectorStore, QdrantConfig

# Using async context manager
async with QdrantVectorStore(QdrantConfig(host="localhost", port=6333)) as store:
    # Add vectors
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata = [{"name": "func1"}, {"name": "func2"}]
    ids = await store.add_vectors("code_entities", vectors, metadata)

    # Search
    results = await store.search_similar([0.1, 0.2, 0.3], top_k=5)

    # Batch search
    batch_results = await store.batch_search(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], top_k=5
    )
```

### Unified Vector Index Manager
```python
from pathlib import Path
from pysearch.storage import VectorIndexManager, EmbeddingConfig

# With OpenAI embeddings + LanceDB
config = EmbeddingConfig(provider="openai", api_key="sk-...")
manager = VectorIndexManager(Path("./db"), config, provider="lancedb")

# Or with HuggingFace + Qdrant
config = EmbeddingConfig(provider="huggingface", model_name="all-MiniLM-L6-v2")
manager = VectorIndexManager(Path("./db"), config, provider="qdrant")

# Or with Chroma
manager = VectorIndexManager(Path("./db"), config, provider="chroma")
```

---

## Dependencies

### Internal Dependencies
- `pysearch.analysis.content_addressing`: IndexTag, content addressing
- `pysearch.indexing.advanced.chunking`: MetadataCodeChunk
- `pysearch.utils.error_handling`: SearchError
- `pysearch.utils.logging_config`: Logger

### External Dependencies (all optional)
- `qdrant-client>=1.7.0`: Qdrant vector database client (AsyncQdrantClient)
- `numpy>=1.24.0`: Numerical operations
- `lancedb`: LanceDB vector database
- `chromadb`: Chroma vector database
- `openai`: OpenAI embeddings API
- `sentence-transformers`: HuggingFace embeddings

---

## Testing

### Integration Tests
Located in `tests/integration/`:
- `test_qdrant_integration.py` - QdrantConfig, QdrantVectorStore, vector utilities tests

---

## Configuration

### Environment Variables
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key
QDRANT_HTTPS=false
```

### SearchConfig Integration
```python
from pysearch.core.config import SearchConfig

config = SearchConfig(
    paths=["./src"],
    qdrant_enabled=True,
    qdrant_host="localhost",
    qdrant_port=6333,
    vector_db_provider="qdrant",  # or "lancedb", "chroma"
    embedding_provider="openai",  # or "huggingface"
)
qdrant_config = config.get_qdrant_config()
```

---

## Performance Considerations

### Batch Operations
- Use `batch_search` for multiple queries in single network call
- Configure `batch_size` based on data size (default: 100)
- All providers support async batch insert

### Compression
- Enable `enable_compression=True` in QdrantConfig for scalar INT8 quantization
- Reduces memory usage at minor accuracy cost

### Resource Cleanup
- Use `async with` context manager for automatic cleanup
- Call `cleanup_orphaned_vectors` periodically to remove stale data

---

## Related Files
- `README.md` - Module overview
- `docs/architecture.md` - Architecture details
- `docs/graphrag_guide.md` - GraphRAG guide with vector storage
- `src/pysearch/indexing/indexes/vector_index.py` - VectorIndex consumer
- `src/pysearch/analysis/graphrag/engine.py` - GraphRAG engine consumer
