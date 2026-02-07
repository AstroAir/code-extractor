# Core Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **core**

---

## Change Log (Changelog)

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive module documentation
- Documented API, configuration, and type system
- Added integration manager descriptions

---

## Module Responsibility

The **Core** module serves as the foundation of PySearch, providing:

1. **Main API**: The `PySearch` class that orchestrates all search operations
2. **Configuration**: The `SearchConfig` class for search parameter management
3. **Type System**: Core data structures (`Query`, `SearchResult`, `SearchItem`, etc.)
4. **History**: Search history tracking and session management
5. **Integration Managers**: Coordination for advanced features

---

## Entry Points and Key Files

### Primary Entry Points
| File | Purpose | Description |
|------|---------|-------------|
| `api.py` | Main API | `PySearch` class - primary search engine interface |
| `config.py` | Configuration | `SearchConfig` class - search parameters and settings |

### Type System Files
| File | Purpose | Description |
|------|---------|-------------|
| `types/__init__.py` | Core Types | `Query`, `SearchResult`, `SearchItem`, `SearchStats` |
| `types/basic_types.py` | Basic Types | `OutputFormat`, `MatchSpan`, `Language`, `FileMetadata` |
| `types/graphrag_types.py` | GraphRAG Types | `GraphRAGQuery`, `GraphRAGResult`, `KnowledgeGraph` |

### History Management Files
| File | Purpose | Description |
|------|---------|-------------|
| `history/__init__.py` | History Interface | `SearchHistory` class - main history interface |
| `history/history_core.py` | Core History | `SearchHistoryEntry` - individual history entries |
| `history/history_sessions.py` | Sessions | Search session tracking |
| `history/history_bookmarks.py` | Bookmarks | Bookmark management |
| `history/history_analytics.py` | Analytics | Search analytics and statistics |

### Integration Managers
| File | Purpose | Description |
|------|---------|-------------|
| `integrations/hybrid_search.py` | Advanced Search | Hybrid search combining traditional, GraphRAG, enhanced indexing |
| `integrations/cache_integration.py` | Cache | File content and result caching |
| `integrations/dependency_integration.py` | Dependencies | Dependency graph analysis |
| `integrations/indexing_integration.py` | Enhanced Indexing | Advanced metadata indexing |
| `integrations/file_watching.py` | File Watching | Real-time file change monitoring |
| `integrations/graphrag_integration.py` | GraphRAG | Knowledge graph construction and querying |
| `integrations/multi_repo_integration.py` | Multi-Repo | Multi-repository search coordination |
| `integrations/parallel_processing.py` | Parallel Processing | Parallel search execution |

---

## PySearch Class (Main API)

### Overview
The `PySearch` class is the primary interface for programmatic access to PySearch functionality. It orchestrates all search operations including indexing, matching, scoring, and output formatting.

### Key Methods

#### Search Methods
```python
# Basic search
search(pattern: str, regex: bool = False, **kwargs) -> SearchResult

# Execute pre-built query
run(query: Query, use_cache: bool = True) -> SearchResult

# Count-only search (fast)
search_count_only(pattern: str, **kwargs) -> CountResult

# Advanced semantic search
search_semantic_advanced(query: str, **kwargs) -> SearchResult

# Hybrid search (combines all methods)
async def hybrid_search(pattern: str, **kwargs) -> dict
```

#### Configuration Methods
```python
# Enable/disable features
enable_caching(**kwargs) -> bool
enable_multi_repo(max_workers: int = 4) -> bool
enable_auto_watch(**kwargs) -> bool

# Get configuration/status
get_cache_stats() -> dict
get_multi_repo_health() -> dict
get_watch_stats() -> dict
```

#### History and Analytics
```python
# History management
get_search_history(limit: int = None) -> list
get_bookmarks() -> dict
add_bookmark(name: str, query: Query, result: SearchResult) -> None

# Analytics
get_search_analytics(days: int = 30) -> dict
get_frequent_patterns(limit: int = 10) -> list
get_pattern_suggestions(partial_pattern: str, limit: int = 5) -> list
```

#### Dependency Analysis
```python
analyze_dependencies(directory: Path = None) -> DependencyGraph
get_dependency_metrics(graph: DependencyGraph = None) -> DependencyMetrics
find_dependency_impact(module: str, graph: DependencyGraph = None) -> dict
suggest_refactoring_opportunities(graph: DependencyGraph = None) -> list
```

---

## SearchConfig Class

### Overview
The `SearchConfig` class defines all search parameters including search scope, behavior, performance settings, and output formatting.

### Key Configuration Areas

#### Search Scope
```python
paths: list[str]              # Root search paths
include: list[str] | None     # Include glob patterns
exclude: list[str] | None     # Exclude glob patterns
languages: set[Language]      # Language filter
file_size_limit: int         # Max file size (default: 2MB)
```

#### Search Behavior
```python
context: int                   # Context lines around matches
output_format: OutputFormat    # Output format (TEXT/JSON/HIGHLIGHT)
follow_symlinks: bool          # Follow symbolic links
enable_docstrings: bool        # Search in docstrings
enable_comments: bool          # Search in comments
enable_strings: bool           # Search in string literals
```

#### Performance Settings
```python
parallel: bool                 # Enable parallel processing
workers: int                   # Worker count (0 = auto)
cache_dir: Path | None         # Cache directory
strict_hash_check: bool        # Use SHA1 for exact change detection
dir_prune_exclude: bool        # Prune excluded dirs during traversal
```

#### Ranking Configuration
```python
rank_strategy: RankStrategy    # Ranking strategy
ast_weight: float              # AST match weight (default: 2.0)
text_weight: float             # Text match weight (default: 1.0)
```

#### Advanced Features
```python
# GraphRAG
enable_graphrag: bool
graphrag_max_hops: int
graphrag_min_confidence: float

# Enhanced Indexing
enable_metadata_indexing: bool
enhanced_indexing_include_semantic: bool

# Qdrant Vector Database
qdrant_enabled: bool
qdrant_host: str
qdrant_port: int
qdrant_collection_name: str
```

---

## Type System

### Core Types

#### Query
```python
@dataclass
class Query:
    pattern: str                      # Search pattern
    use_regex: bool = False           # Enable regex
    use_ast: bool = False             # Enable AST matching
    use_semantic: bool = False        # Enable semantic search
    use_boolean: bool = False         # Enable boolean logic
    context: int = 2                  # Context lines
    ast_filters: ASTFilters | None    # AST filters
    metadata_filters: MetadataFilters | None  # Metadata filters
    output: OutputFormat = OutputFormat.TEXT
    count_only: bool = False          # Count-only mode
    max_per_file: int | None = None   # Max results per file
```

#### SearchResult
```python
@dataclass
class SearchResult:
    items: list[SearchItem]           # Match results
    stats: SearchStats                # Search statistics
```

#### SearchItem
```python
@dataclass
class SearchItem:
    file: Path                        # File path
    start_line: int                   # Start line
    end_line: int                     # End line
    lines: list[str]                  # Matched lines
    match_spans: list[MatchSpan]      # Match positions
    score: float = 0.0                # Relevance score
```

#### ASTFilters
```python
@dataclass
class ASTFilters:
    func_name: str | None = None      # Function name pattern
    class_name: str | None = None     # Class name pattern
    decorator: str | None = None      # Decorator pattern
    imported: str | None = None       # Import pattern
```

#### MetadataFilters
```python
@dataclass
class MetadataFilters:
    min_size: str | None = None       # Minimum file size
    max_size: str | None = None       # Maximum file size
    min_date: str | None = None       # Minimum modification date
    max_date: str | None = None       # Maximum modification date
    languages: set[Language] | None   # Language filter
    author_pattern: str | None = None # Author pattern
```

---

## Integration Managers

### Advanced Search Manager
Coordinates hybrid search combining traditional search, GraphRAG, and enhanced indexing.

### Cache Integration Manager
Manages file content caching and result caching with TTL-based expiration.

### Dependency Integration Manager
Handles dependency graph analysis, circular dependency detection, and refactoring suggestions.

### Enhanced Indexing Integration Manager
Manages advanced metadata indexing with complexity analysis and dependency tracking.

### File Watching Manager
Provides real-time file change monitoring with debouncing and batch processing.

### GraphRAG Integration Manager
Handles knowledge graph construction, querying, and vector database integration.

### Multi-Repo Integration Manager
Coordinates search across multiple repositories with parallel execution.

### Parallel Processing Manager
Manages parallel search execution with configurable worker pools.

---

## Dependencies

### Internal Dependencies
- `pysearch.indexing`: File indexing and caching
- `pysearch.search`: Pattern matching algorithms
- `pysearch.utils`: Utility functions and error handling
- `pysearch.storage`: Vector database integration

### External Dependencies
- `pydantic`: Configuration validation
- `click`: CLI framework (for CLI module)
- `rich`: Terminal output formatting
- `orjson`: Fast JSON serialization

---

## Testing

### Unit Tests
Located in `tests/unit/core/`:
- API tests: `test_api_*.py`
- Configuration tests: `test_config.py`
- Type system tests: `test_types.py`
- History tests: `test_history_*.py`

### Integration Tests
Located in `tests/integration/`:
- Advanced search integration
- Multi-repo integration
- GraphRAG integration
- Enhanced indexing integration

---

## Common Usage Patterns

### Basic Search
```python
from pysearch import PySearch, SearchConfig

config = SearchConfig(paths=["."], include=["**/*.py"])
engine = PySearch(config)
results = engine.search("def main")
```

### Advanced Query
```python
from pysearch.types import Query, ASTFilters

filters = ASTFilters(func_name=".*handler", decorator="lru_cache")
query = Query(pattern="def", use_ast=True, ast_filters=filters)
results = engine.run(query)
```

### With History and Bookmarks
```python
# Get search history
history = engine.get_search_history(limit=10)

# Bookmark a search
engine.add_bookmark("my_search", query, results)

# Get analytics
analytics = engine.get_search_analytics(days=30)
```

---

## Related Files
- `README.md` - Project overview
- `docs/architecture.md` - Detailed architecture
- `docs/api/pysearch.md` - API reference
- `tests/conftest.py` - Test fixtures
