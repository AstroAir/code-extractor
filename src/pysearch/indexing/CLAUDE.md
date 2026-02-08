# Indexing Module

[根目录](../../../CLAUDE.md) > **indexing**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Updated with advanced indexing engine documentation
- Added cache system documentation with backends
- Enhanced metadata indexing documentation
- Synchronized with current project structure

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive indexing module documentation

---

## Module Responsibility

The **Indexing** module provides file scanning, caching, metadata management, and advanced indexing capabilities for the PySearch engine.

### Key Responsibilities
1. **File Indexing**: Efficient file discovery and scanning
2. **Cache Management**: Multi-level caching with pluggable backends
3. **Metadata Indexing**: Content-addressed metadata storage
4. **Advanced Indexing**: Chunking, distributed processing, and coordination
5. **Specialized Indexes**: Full-text, vector, chunk, and code snippet indexes

---

## Entry and Startup

### Main Entry Points
- **`indexer.py`** - Basic file indexer
  - `Indexer.__init__(config)` - Initialize indexer
  - `Indexer.scan()` - Scan for files
  - `Indexer.load/save()` - Load/save cache

- **`advanced/engine.py`** - Advanced indexing engine
  - `EnhancedIndexingEngine.index_codebase()` - Index codebase
  - `EnhancedIndexingEngine.search_index()` - Search index

- **`cache/manager.py`** - Cache manager
  - `CacheManager.get/set()` - Cache operations
  - `CacheManager.clear()` - Clear cache

---

## Public API

### Basic Indexing

```python
from pysearch.indexing import Indexer
from pysearch import SearchConfig

config = SearchConfig(paths=["."], include=["**/*.py"])
indexer = Indexer(config)

# Scan for files
changed, removed, total = indexer.scan()

# Iterate files
for file_path in indexer.iter_files():
    print(file_path)

# Save index
indexer.save()
```

### Advanced Indexing

```python
from pysearch.indexing.advanced import EnhancedIndexingEngine

engine = EnhancedIndexingEngine(config)

# Index codebase
await engine.index_codebase(directories=["./src"])

# Search index
results = await engine.search_index(query="database")
```

### Cache Management

```python
from pysearch.indexing.cache import CacheManager

cache = CacheManager(backend="memory", max_size=1000)

# Set/get cache
cache.set("key", {"data": "value"}, ttl=3600)
result = cache.get("key")

# Clear cache
cache.clear()
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.core.config` - Configuration
- `pysearch.utils.helpers` - File utilities
- `pysearch.analysis.language_detection` - Language detection

### External Dependencies
- No special external dependencies for basic indexing
- Advanced indexing may require: `qdrant-client`, `sentence-transformers`

---

## Data Models

### Indexer Models
- `IndexRecord` - File metadata record (path, size, mtime, sha1)

### Cache Models (`cache/models.py`)
- `CacheEntry` - Cache entry with metadata
- `CacheStats` - Cache statistics
- `CacheConfig` - Cache configuration

### Metadata Models (`metadata/models.py`)
- `FileMetadata` - Extended file metadata
- `EntityMetadata` - Code entity metadata
- `DependencyMetadata` - Dependency information

---

## Testing

### Test Directory
- `tests/unit/indexing/` - Indexing module tests
  - `test_indexer.py` - Basic indexer tests
  - `cache/` - Cache system tests
  - `metadata/` - Metadata indexing tests
  - `advanced/` - Advanced indexing tests
  - `indexes/` - Index tests

### Running Tests
```bash
pytest tests/unit/indexing/ -v
pytest tests/unit/indexing/cache/ -v
pytest tests/unit/indexing/advanced/ -v
```

---

## Common Issues and Solutions

### Issue 1: Cache not persisting
**Symptoms**: Cache lost between runs
**Solution**: Ensure cache directory is writable and permissions are correct

### Issue 2: Memory usage too high
**Symptoms**: High memory during indexing
**Solution**: Use disk cache backend or reduce cache size:
```python
config.cache_dir = "./.pysearch-cache"
cache = CacheManager(backend="disk", max_size=500)
```

### Issue 3: Indexing slow on large codebases
**Symptoms**: Indexing takes too long
**Solution**: Enable parallel processing:
```python
config.parallel = True
config.workers = 4  # or 0 for auto-detect
```

---

## Related Files

### Indexing Module Files
- `src/pysearch/indexing/__init__.py`
- `src/pysearch/indexing/indexer.py` - Basic indexer
- `src/pysearch/indexing/cache/` - Cache system (6 files)
- `src/pysearch/indexing/metadata/` - Metadata indexing (4 files)
- `src/pysearch/indexing/advanced/` - Advanced indexing (6 files)
- `src/pysearch/indexing/indexes/` - Specialized indexes (4 files)

---

## Module Structure

```
indexing/
├── __init__.py
├── indexer.py                    # Basic file indexer
├── cache/                        # Cache system
│   ├── __init__.py
│   ├── manager.py               # CacheManager
│   ├── backends.py              # Cache backends (memory, disk)
│   ├── models.py                # Cache data models
│   ├── dependencies.py          # Cache dependency tracking
│   ├── cleanup.py               # Cache cleanup
│   └── statistics.py            # Cache statistics
├── metadata/                     # Metadata indexing
│   ├── __init__.py
│   ├── indexer.py               # MetadataIndexer
│   ├── database.py              # Metadata database
│   ├── models.py                # Metadata models
│   └── analysis.py              # Metadata analysis
├── advanced/                     # Advanced indexing
│   ├── __init__.py
│   ├── engine.py                # EnhancedIndexingEngine
│   ├── coordinator.py           # IndexCoordinator
│   ├── base.py                  # Base classes
│   ├── locking.py               # Index locking
│   ├── chunking.py              # Code chunking
│   └── integration.py           # Integration helpers
└── indexes/                      # Specialized indexes
    ├── __init__.py
    ├── full_text_index.py       # Full-text search
    ├── vector_index.py          # Vector similarity
    ├── chunk_index.py           # Chunk-based index
    └── code_snippets_index.py   # Code snippets
```
