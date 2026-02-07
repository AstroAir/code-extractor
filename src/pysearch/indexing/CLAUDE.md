# Indexing Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **indexing**

---

## Change Log (Changelog)

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive module documentation
- Documented indexer, cache, metadata, and advanced indexing
- Added refactoring notes from 2025-08-15

---

## Module Responsibility

The **Indexing** module handles file discovery, metadata extraction, caching, and advanced indexing capabilities:

1. **File Indexing**: Efficient file scanning with incremental updates
2. **Caching**: Performance optimization through intelligent caching
3. **Metadata Management**: File metadata indexing and querying
4. **Advanced Features**: Advanced indexing with content addressing and multi-index architecture

---

## Entry Points and Key Files

### Primary Entry Points
| File | Purpose | Description |
|------|---------|-------------|
| `indexer.py` | Basic Indexer | `Indexer` class - file scanning and incremental indexing |
| `cache_manager.py` | Cache Manager | Backward compatibility for cache management |
| `metadata.py` | Metadata Interface | Backward compatibility for metadata indexing |

### Cache Package
| File | Purpose | Description |
|------|---------|-------------|
| `cache/__init__.py` | Cache Interface | `CacheManager` class - main cache interface |
| `cache/manager.py` | Cache Manager | Cache implementation with backends |
| `cache/backends.py` | Cache Backends | Memory and disk cache implementations |
| `cache/dependencies.py` | Dependencies | Dependency-based cache invalidation |
| `cache/cleanup.py` | Cleanup | Cache cleanup and eviction policies |
| `cache/statistics.py` | Statistics | Cache performance statistics |

### Metadata Package
| File | Purpose | Description |
|------|---------|-------------|
| `metadata/__init__.py` | Metadata Interface | Backward compatibility imports |
| `metadata/models.py` | Data Models | Metadata data models |
| `metadata/database.py` | Database | SQLite database for metadata |
| `metadata/indexer.py` | Metadata Indexer | Entity-level indexing |
| `metadata/analysis.py` | Analysis | Metadata analysis utilities |

### Advanced Package
| File | Purpose | Description |
|------|---------|-------------|
| `advanced/__init__.py` | Advanced Interface | MetadataCodeChunkg exports |
| `advanced/engine.py` | Engine | `IndexingEngine` - main advanced indexing engine |
| `advanced/coordinator.py` | Coordinator | `IndexCoordinator` - multi-index coordination |
| `advanced/base.py` | Base Classes | `CodebaseIndex` - abstract base for indexes |
| `advanced/locking.py` | Locking | `IndexLock` - concurrent indexing prevention |
| `advanced/chunking.py` | Chunking | Code-aware chunking strategies |
| `advanced/integration.py` | Integration | `IndexSearchEngine` - high-level API |

### Indexes Package
| File | Purpose | Description |
|------|---------|-------------|
| `indexes/__init__.py` | Indexes Interface | Index exports |
| `indexes/code_snippets_index.py` | Code Snippets | Tree-sitter based entity extraction |
| `indexes/full_text_index.py` | Full-Text | SQLite FTS5 full-text search |
| `indexes/chunk_index.py` | Chunks | Intelligent code chunking |
| `indexes/vector_index.py` | Vector | Vector database integration |

### Legacy Files
| File | Purpose | Description |
|------|---------|-------------|
| `legacy_indexer.py` | Legacy Indexer | Deprecated backward compatibility |

---

## Recent Refactoring (2025-08-15)

This module was refactored to improve maintainability and follow the single responsibility principle:

- **Metadata Package**: Split 848-line file into focused modules
- **Advanced Package**: Split 689-line file into specialized modules
- **Cache Package**: Refactored 421-line file with better separation

All files now follow the 300-line coding standard while maintaining backward compatibility.

---

## Indexer Class (Basic Indexing)

### Overview
The `Indexer` class provides basic file indexing with incremental updates based on file modification times and optional hash checking.

### Key Methods
```python
class Indexer:
    def __init__(self, config: SearchConfig)
    def scan(self) -> tuple[list[Path], list[Path], int]  # Scan for changes
    def iter_files(self) -> Iterator[Path]                 # Iterate files
    def iter_all_paths(self) -> Iterator[Path]             # All indexed paths
    def save(self) -> None                                 # Save index
    def count_indexed(self) -> int                         # Count indexed files
    def get_cache_stats(self) -> dict                      # Cache statistics
    def cleanup_old_entries(self, days_old: int = 30) -> int
```

### Indexing Strategy
1. **Incremental Updates**: Only re-index changed files based on mtime/size
2. **Optional Hash Check**: SHA1 for exact change detection (if enabled)
3. **Directory Pruning**: Skip excluded directories during traversal
4. **Cache Persistence**: JSON-based cache storage

---

## CacheManager Class

### Overview
The `CacheManager` class provides multi-level caching with configurable backends and eviction policies.

### Key Methods
```python
class CacheManager:
    def __init__(self, backend: str = "memory", cache_dir: Path = None, ...)
    def get(self, key: str) -> Any | None
    def set(self, key: str, value: Any, ttl: float | None = None) -> None
    def invalidate(self, key: str) -> bool
    def invalidate_by_file(self, file_path: str) -> int
    def clear(self) -> None
    def get_stats(self) -> dict
    def shutdown(self) -> None
```

### Cache Backends
- **Memory**: In-memory caching with LRU eviction
- **Disk**: Persistent disk-based caching with compression

---

## Advanced Indexing Engine

### Overview
The `IndexingEngine` provides sophisticated indexing capabilities with content addressing, tag-based management, and multi-index architecture.

### Key Features

#### Content-Addressed Caching
- SHA256-based content addressing for efficient deduplication
- Global cache for cross-branch content sharing
- Smart diffing with compute/delete/addTag/removeTag operations

#### Tag-Based Index Management
- Three-part tag system: directory + branch + artifact
- Multi-branch support without reindexing
- Granular index management and cleanup

#### Multi-Index Architecture
- **Code Snippets Index**: Tree-sitter based entity extraction
- **Full-Text Search Index**: SQLite FTS5 with trigram tokenization
- **Chunk Index**: Intelligent code-aware chunking
- **Vector Index**: Vector database integration for semantic search

### Key Classes
```python
class CodebaseIndex(ABC):
    @property
    @abstractmethod
    def artifact_id(self) -> str

    @property
    @abstractmethod
    def relative_expected_time(self) -> float

    @abstractmethod
    async def update(self, tag: IndexTag, results: RefreshIndexResults, ...) -> None

class IndexingEngine:
    async def refresh_index(self, force: bool = False) -> bool
    async def get_progress(self) -> IndexingProgress
    def cancel_indexing(self) -> None
```

---

## Chunking Strategies

### Overview
The chunking system provides code-aware content segmentation for better semantic indexing.

### Chunking Types
- **Structural Chunker**: AST/tree-sitter based structure-aware chunking
- **Semantic Chunker**: Semantic-aware chunking for content organization
- **Hybrid Chunker**: Combines structural and semantic approaches

### Usage
```python
from pysearch.indexing.advanced.chunking import ChunkingEngine, ChunkingConfig

config = ChunkingConfig(strategy="hybrid", max_chunk_size=1500)
engine = ChunkingEngine(config)
chunks = await engine.chunk_file("example.py", content)
```

---

## Metadata Indexing

### Overview
The metadata indexing system provides entity-level indexing with SQLite storage.

### Key Components
- **Database**: SQLite storage for metadata
- **Models**: Data models for entities and relationships
- **Indexer**: Entity extraction and indexing
- **Analysis**: Metadata analysis utilities

### Capabilities
- Entity extraction (functions, classes, methods)
- Relationship tracking (imports, dependencies)
- Complexity analysis
- Dependency graph construction

---

## Specialized Indexes

### Code Snippets Index
Extracts top-level code structures using tree-sitter queries for multiple languages.

### Full-Text Search Index
Provides fast text-based search using SQLite FTS5 with trigram tokenization.

### Chunk Index
Intelligent code chunking with language-aware strategies for embeddings.

### Vector Index
Vector database integration for semantic similarity search with support for multiple backends.

---

## Dependencies

### Internal Dependencies
- `pysearch.core`: Configuration and types
- `pysearch.utils`: File utilities and error handling
- `pysearch.analysis`: Language detection and analysis

### External Dependencies
- `pathspec`: Glob pattern matching
- `aiofiles`: Async file operations
- `sqlite3`: Database (built-in)

---

## Testing

### Unit Tests
Located in `tests/unit/core/`:
- `test_indexer_*.py` - Indexer tests
- `test_cache_manager_*.py` - Cache tests
- `test_metadata_*.py` - Metadata tests

### Integration Tests
Located in `tests/integration/`:
- `tenable_metadata_indexing.py` - Advanced indexing tests

---

## Common Usage Patterns

### Basic Indexing
```python
from pysearch import PySearch, SearchConfig

config = SearchConfig(paths=["."], include=["**/*.py"])
engine = PySearch(config)

# Indexer scans automatically on first search
results = index_search("def main")
```

### Advanced Indexing
```python
from pysearch.indexing.advanced.engine import IndexingEngine

engine = IndexingEngine(config)
await engine.refresh_index(force=False)
```

### Cache Management
```python
from pysearch.indexing.cache import CacheManager

cache = CacheManager(backend="disk", cache_dir="/tmp/cache")
cache.set("key", data, ttl=3600)
result = cache.get("key")
```

---

## Related Files
- `README.md` - Module overview
- `docs/indexing-architecture.md` - Detailed indexing architecture
- `docs/advanced-indexing-guide.md` - Advanced indexing guide
