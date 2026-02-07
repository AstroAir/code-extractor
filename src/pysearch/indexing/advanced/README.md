# Advanced Indexing

This directory contains advanced indexing functionality that extends the basic indexing system with sophisticated features for enterprise-grade search capabilities.

## Overview

The advanced indexing system provides:

- **Indexing engine** with advanced algorithms and optimizations
- **Seamless integration** with the main search system
- **Advanced code-aware chunking** strategies for better content organization
- **Performance monitoring** and health checks
- **Multi-index coordination** for comprehensive search coverage

## Components

### Core Engine (`engine.py`)

- `CodebaseIndex`: Abstract base class for all advanced index types
- `IndexCoordinator`: Coordinates multiple index types for unified search
- `IndexingEngine`: Main indexing engine with tag-based management
- `IndexLock`: Prevents concurrent indexing operations

### Integration (`integration.py`)

- `IndexSearchEngine`: Main search engine integrating all advanced features
- `SearchResultEnhancer`: Enhances search results with additional metadata
- `IndexingOrchestrator`: Orchestrates indexing operations across repositories

### Advanced Chunking (`chunking.py`)

- `ChunkingEngine`: Main chunking coordination engine
- `StructuralChunker`: Structure-aware chunking using AST/tree-sitter
- `SemanticChunker`: Semantic-aware chunking for better content organization
- `HybridChunker`: Combines multiple chunking approaches

## Features

### Content-Addressed Caching

- SHA256-based content addressing for efficient incremental updates
- Global cache for cross-branch content sharing
- Smart diffing with compute/delete/addTag/removeTag operations

### Tag-Based Index Management

- Three-part tag system: directory + branch + artifact
- Multi-branch support without reindexing
- Granular index management and cleanup

### Multi-Index Architecture

- Code Snippets Index: Tree-sitter based entity extraction
- Full-Text Search Index: SQLite FTS5 with trigram tokenization
- Chunk Index: Intelligent code-aware chunking
- Vector Index: Vector database integration for semantic search

## Usage

### Basic Enhanced Indexing

```python
from pysearch.indexing.advanced.engine import IndexingEngine
from pysearch.core.config import SearchConfig

config = SearchConfig(paths=["./src"])
engine = IndexingEngine(config)
await engine.refresh_index()
```

### Advanced Search

```python
from pysearch.indexing.advanced.integration import IndexSearchEngine

engine = IndexSearchEngine(config)
results = await engine.search("database connection", limit=10)
```

### Custom Chunking

```python
from pysearch.indexing.advanced.chunking import ChunkingEngine, ChunkingConfig

config = ChunkingConfig(strategy="hybrid", max_chunk_size=1500)
engine = ChunkingEngine(config)
chunks = await engine.chunk_file("example.py", content)
```

## Architecture

The advanced indexing system follows a modular architecture:

1. **Engine Layer**: Core indexing algorithms and coordination
2. **Integration Layer**: High-level APIs and search orchestration  
3. **Chunking Layer**: Content organization and segmentation
4. **Storage Layer**: Persistent storage and caching mechanisms

## Performance

The system is designed for high performance with:

- Parallel processing capabilities
- Incremental updates to minimize reindexing
- Content addressing for efficient caching
- Optimized data structures for fast retrieval

## Compatibility

This module maintains backward compatibility with the existing pysearch APIs while providing enhanced functionality through new interfaces.
