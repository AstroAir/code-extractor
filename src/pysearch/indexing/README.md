# Indexing Module

The indexing module handles file indexing, caching, and metadata management for efficient search operations.

## Recent Refactoring (2025-08-15)

This module has been refactored to improve maintainability and follow the single responsibility principle. Large files have been broken down into focused, modular components while preserving all existing functionality and public APIs.

### Refactored Components

- **Metadata Package** (`metadata/`): Split 848-line file into focused modules
- **Advanced Package** (`advanced/`): Split 689-line file into specialized modules
- **Cache Package** (`cache/`): Refactored 421-line file with better separation of concerns

All files now follow the 300-line coding standard while maintaining backward compatibility.

## Responsibilities

- **File Indexing**: Efficient file scanning and metadata extraction
- **Caching**: Performance optimization through intelligent caching
- **Metadata Management**: File metadata indexing and querying
- **Advanced Features**: Advanced indexing capabilities (in `advanced/` subdirectory)

## Key Files

- `indexer.py` - Basic file indexing with incremental updates (291 lines)
- `metadata.py` - Backward compatibility module importing from `metadata/` package
- `metadata/` - Modular metadata indexing system (models, database, indexer, analysis)
- `cache_manager.py` - Backward compatibility for cache management
- `cache/` - Modular cache management package (manager, dependencies, cleanup, statistics)
- `advanced/` - Modular advanced indexing (base, locking, coordinator, engine)
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
