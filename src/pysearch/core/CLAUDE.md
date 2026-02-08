# Core Module

[根目录](../../../CLAUDE.md) > **core**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Updated with latest refactored structure
- Added integration managers documentation
- Enhanced type system and history documentation
- Synchronized with current project state

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive core module documentation

---

## Module Responsibility

The **Core** module provides the main API, configuration system, type definitions, search history tracking, and integration managers for the PySearch engine.

### Key Responsibilities
1. **Main API**: `PySearch` class - primary entry point for search operations
2. **Configuration**: `SearchConfig` - centralized configuration management
3. **Type System**: Core data types for queries, results, filters
4. **History**: Search history, bookmarks, analytics, and sessions
5. **Integration Managers**: Modular managers for advanced features

---

## Entry and Startup

### Main Entry Points
- **`api.py`** - Main `PySearch` class (1732 lines)
  - `PySearch.__init__(config)` - Initialize search engine
  - `PySearch.search(pattern)` - Execute text search
  - `PySearch.run(query)` - Run custom query
  - `PySearch.search_semantic(query)` - Semantic search

- **`config.py`** - Configuration classes (322 lines)
  - `SearchConfig` - Main configuration dataclass
  - `RankStrategy` - Ranking strategy enum

- **`__init__.py`** - Public API exports
  - Exports: `PySearch`, `SearchConfig`, `Query`, `SearchResult`, etc.

---

## Public API

### PySearch Class

```python
from pysearch import PySearch, SearchConfig

# Initialize with configuration
config = SearchConfig(paths=["."], include=["**/*.py"])
engine = PySearch(config)

# Basic search
results = engine.search("def main")

# Advanced search with Query
from pysearch.core.types import Query
query = Query(pattern="def", use_regex=True, context=3)
results = engine.run(query)

# Semantic search
results = engine.search_semantic("database connection", threshold=0.7)
```

### SearchConfig Class

```python
from pysearch import SearchConfig

config = SearchConfig(
    paths=["./src", "./tests"],
    include=["**/*.py"],
    exclude=["**/.venv/**"],
    context=3,
    parallel=True,
    workers=4,
    strict_hash_check=False,
    enable_graphrag=True,
    qdrant_enabled=True
)
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.indexing.indexer` - File indexing
- `pysearch.search.matchers` - Pattern matching
- `pysearch.search.scorer` - Result ranking
- `pysearch.utils.formatter` - Output formatting

### External Dependencies
- `click>=8.1.7` - CLI framework
- `pydantic>=2.7.0` - Config validation
- `rich>=13.7.1` - Terminal output
- `orjson>=3.10.7` - JSON serialization

### Configuration Files
- `pyproject.toml` - Package configuration, dependencies, tool settings
- `configs/config.example.toml` - Example configuration file

---

## Data Models

### Core Types (`types/`)

#### Basic Types (`basic_types.py`)
- `Language` - Supported programming languages enum (26 languages)
- `OutputFormat` - Output format enum (TEXT, JSON, HIGHLIGHT)
- `ASTFilters` - AST-based filter configuration
- `MetadataFilters` - File metadata filter configuration
- `Query` - Search query specification
- `SearchItem` - Individual search result
- `SearchResult` - Complete search results with statistics
- `SearchStats` - Performance statistics

#### GraphRAG Types (`graphrag_types.py`)
- `CodeEntity` - Code entity representation
- `EntityType` - Entity type enum (FUNCTION, CLASS, VARIABLE, etc.)
- `EntityRelationship` - Relationship between entities
- `RelationType` - Relation type enum (IMPORTS, CALLS, INHERITS, etc.)
- `KnowledgeGraph` - Knowledge graph structure
- `GraphRAGQuery` - GraphRAG query specification
- `GraphRAGResult` - GraphRAG search result

### History Types (`history/`)

#### History Core (`history_core.py`)
- `SearchHistory` - Search history manager
- `HistoryEntry` - Single history entry

#### History Sessions (`history_sessions.py`)
- `SessionManager` - Search session management
- `SearchSession` - Session data structure

#### History Bookmarks (`history_bookmarks.py`)
- `BookmarkManager` - Bookmark management
- `BookmarkFolder` - Bookmark folder organization
- `BookmarkEntry` - Single bookmark entry

#### History Analytics (`history_analytics.py`)
- `AnalyticsManager` - Search analytics
- `SearchCategory` - Search category enum
- `PerformanceMetrics` - Performance metrics data

---

## Testing

### Test Directory
- `tests/unit/core/` - Core module tests
  - `test_api.py` - PySearch API tests
  - `test_config.py` - Configuration tests
  - `test_types/` - Type system tests
  - `test_history/` - History system tests
  - `test_managers/` - Integration manager tests

### Running Tests
```bash
# Run all core tests
pytest tests/unit/core/ -v

# Run specific test file
pytest tests/unit/core/test_api.py -v

# Run with coverage
pytest tests/unit/core/ --cov=src/pysearch/core
```

---

## Common Issues and Solutions

### Issue 1: Configuration validation errors
**Symptoms**: `ConfigurationError` on initialization
**Solution**: Check that required fields are provided and optional features are properly configured:
```python
config = SearchConfig(paths=["."])  # paths is required
config.validate()  # Explicit validation
```

### Issue 2: Search history not persisting
**Symptoms**: History lost between sessions
**Solution**: Ensure cache directory is writable:
```python
config = SearchConfig(paths=["."], cache_dir="./.pysearch-cache")
```

### Issue 3: Type checking errors
**Symptoms**: mypy errors on core types
**Solution**: Ensure type stubs are installed:
```bash
pip install types-regex
mypy src/pysearch/core/
```

---

## Related Files

### Core Module Files
- `src/pysearch/core/__init__.py` - Package initialization
- `src/pysearch/core/api.py` - Main PySearch API
- `src/pysearch/core/config.py` - Configuration classes
- `src/pysearch/core/types/basic_types.py` - Basic type definitions
- `src/pysearch/core/types/graphrag_types.py` - GraphRAG types
- `src/pysearch/core/types/__init__.py` - Type exports
- `src/pysearch/core/history/*.py` - History system (4 files)
- `src/pysearch/core/managers/*.py` - Integration managers (10 files)

### Test Files
- `tests/unit/core/test_api.py` - API tests
- `tests/unit/core/test_config.py` - Config tests
- `tests/unit/core/types/` - Type tests
- `tests/unit/core/history/` - History tests
- `tests/unit/core/managers/` - Manager tests

---

## FAQ

### Q: How do I enable GraphRAG?
A: Set `enable_graphrag=True` and ensure Qdrant is configured:
```python
config = SearchConfig(
    paths=["."],
    enable_graphrag=True,
    qdrant_enabled=True,
    qdrant_host="localhost",
    qdrant_port=6333
)
```

### Q: How do I customize ranking?
A: Use the `search_with_ranking` method or set ranking weights:
```python
config.ast_weight = 2.0
config.text_weight = 1.0
results = engine.search_with_ranking("def main", ranking_strategy="relevance")
```

### Q: How do I access search history?
A: Use the history property:
```python
history = engine.get_search_history(limit=20)
analytics = engine.get_search_analytics(days=30)
```

### Q: How do I enable multi-repo search?
A: Use the integration manager:
```python
engine.enable_multi_repo(max_workers=4)
engine.add_repository("frontend", "./frontend")
engine.add_repository("backend", "./backend")
results = engine.search_all_repositories("async def")
```

---

## Module Structure

```
core/
├── __init__.py              # Public API exports
├── api.py                   # Main PySearch class
├── config.py                # SearchConfig and RankStrategy
├── types/                   # Type definitions
│   ├── __init__.py
│   ├── basic_types.py       # Query, SearchResult, SearchItem, etc.
│   └── graphrag_types.py    # GraphRAG-specific types
├── history/                 # Search history system
│   ├── __init__.py
│   ├── history_core.py      # SearchHistory manager
│   ├── history_sessions.py  # Session management
│   ├── history_bookmarks.py # Bookmark management
│   └── history_analytics.py # Analytics manager
└── managers/                # Integration managers
    ├── __init__.py
    ├── hybrid_search.py             # Semantic search integration
    ├── graphrag_integration.py      # GraphRAG integration
    ├── ide_integration.py            # IDE hooks integration
    ├── distributed_indexing_integration.py  # Distributed indexing
    ├── multi_repo_integration.py     # Multi-repo search
    ├── dependency_integration.py     # Dependency analysis
    ├── file_watching.py              # File watching
    ├── cache_integration.py          # Cache management
    ├── indexing_integration.py       # Metadata indexing
    └── parallel_processing.py        # Parallel search
```
