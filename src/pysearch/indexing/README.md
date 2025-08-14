# Indexing Module

The indexing module handles file indexing, caching, and metadata management for efficient search operations.

## Responsibilities

- **File Indexing**: Efficient file scanning and metadata extraction
- **Caching**: Performance optimization through intelligent caching
- **Metadata Management**: File metadata indexing and querying
- **Advanced Features**: Advanced indexing capabilities (in `advanced/` subdirectory)

## Key Files

- `indexer.py` - Basic file indexing with incremental updates
- `metadata.py` - Metadata indexing and querying (renamed from `indexer_metadata.py`)
- `cache_manager.py` - Caching system for performance optimization
- `advanced/` - Advanced indexing features and algorithms
- `legacy_indexer.py` - Backward compatibility module (deprecated)
- `indexes/` - Specialized index implementations

## Indexing Strategy

1. **Incremental Updates**: Only re-index changed files based on mtime/size
2. **Metadata Tracking**: Store file metadata for efficient filtering
3. **Parallel Processing**: Multi-threaded indexing for large codebases
4. **Cache Management**: TTL-based caching with memory optimization

## Advanced Features

The `advanced/` subdirectory contains advanced indexing capabilities:

- Code-aware chunking strategies
- Advanced indexing engine with sophisticated algorithms
- Performance monitoring and optimization

## Usage

```python
from pysearch.indexing import Indexer, MetadataIndexer
from pysearch.core import SearchConfig

config = SearchConfig(paths=["."], include=["**/*.py"])
indexer = Indexer(config)
files = list(indexer.iter_files())
```
