# API Reference

This document provides comprehensive API documentation for pysearch, covering all major classes, functions, and types.

## Quick Navigation

- [PySearch Class](#pysearch-class) - Main search engine
- [SearchConfig Class](#searchconfig-class) - Configuration management
- [Query and Result Types](#query-and-result-types) - Data structures
- [Filters and Metadata](#filters-and-metadata) - Advanced filtering
- [GraphRAG API](#graphrag-api) - Knowledge graph search
- [Fuzzy Search API](#fuzzy-search-api) - Typo-tolerant search
- [Boolean Search API](#boolean-search-api) - Logical query composition
- [Dependency Analysis API](#dependency-analysis-api) - Import graph analysis
- [Multi-Repository API](#multi-repository-api) - Cross-repo search
- [Storage API](#storage-api) - Vector database and embeddings
- [IDE Integration API](#ide-integration-api) - LSP-like features
- [Utility Functions](#utility-functions) - Helper functions
- [Exception Classes](#exception-classes) - Error handling

---

## PySearch Class

The main entry point for all search operations.

### Class Definition

```python
class PySearch:
    """
    Main search engine class for pysearch.
    
    Orchestrates all search operations including file indexing, content
    matching, result scoring, and output formatting.
    """
```

### Constructor

```python
def __init__(self, config: SearchConfig | None = None, logger: SearchLogger | None = None)
```

**Parameters:**

- `config` (SearchConfig, optional): Configuration object. Uses default if None.
- `logger` (SearchLogger, optional): Custom logger instance. Uses default if None.

**Example:**

```python
from pysearch import PySearch, SearchConfig

# Basic initialization
engine = PySearch()

# With custom configuration
config = SearchConfig(paths=["./src"], context=5)
engine = PySearch(config)
```

### Core Methods

#### `search(pattern, regex=False, context=None, output=None, **kwargs)`

High-level search method with convenient parameters.

**Parameters:**

- `pattern` (str): Search pattern (text or regex)
- `regex` (bool): Enable regex matching (default: False)
- `context` (int, optional): Context lines around matches
- `output` (OutputFormat, optional): Output format override
- `**kwargs`: Additional options (use_ast, filters, metadata_filters)

**Returns:** `SearchResult` - Complete search results with statistics

**Example:**

```python
# Simple text search
results = engine.search("def main")

# Regex search with context
results = engine.search(r"def \w+_handler", regex=True, context=5)

# AST search with filters
from pysearch.types import ASTFilters
filters = ASTFilters(func_name="main")
results = engine.search("def", use_ast=True, filters=filters)
```

#### `run(query)`

Execute a complete search query with full control.

**Parameters:**

- `query` (Query): Complete query specification

**Returns:** `SearchResult` - Complete search results with statistics

**Example:**

```python
from pysearch.types import Query, ASTFilters, OutputFormat

query = Query(
    pattern="class.*Test",
    use_regex=True,
    use_ast=True,
    context=3,
    output=OutputFormat.JSON,
    ast_filters=ASTFilters(class_name=".*Test")
)
results = engine.run(query)
```

### Advanced Features

#### `enable_caching(cache_dir=None, ttl=3600)`

Enable result caching for improved performance.

**Parameters:**

- `cache_dir` (Path, optional): Cache directory path
- `ttl` (int): Time-to-live in seconds (default: 3600)

**Example:**

```python
engine.enable_caching(ttl=7200)  # 2 hour cache
```

#### `enable_auto_watch()`

Enable automatic file watching for real-time updates.

**Example:**

```python
engine.enable_auto_watch()
# Engine will automatically update index when files change
```

#### `enable_multi_repo(repositories)`

Enable multi-repository search capabilities.

**Parameters:**

- `repositories` (list[RepositoryInfo]): Repository configurations

**Example:**

```python
from pysearch.multi_repo import RepositoryInfo

repos = [
    RepositoryInfo(name="main", path="./", priority=1.0),
    RepositoryInfo(name="lib", path="../lib", priority=0.8)
]
engine.enable_multi_repo(repos)
```

### Properties

#### `history`

Access to search history tracking.

```python
# Get recent searches
recent = engine.history.get_recent(limit=10)

# Get search statistics
stats = engine.history.get_stats()
```

#### `indexer`

Access to the file indexer for advanced operations.

```python
# Force reindex
engine.indexer.rebuild_index()

# Get index statistics
stats = engine.indexer.get_stats()
```

---

## SearchConfig Class

Configuration management for search operations.

### Class Definition

```python
@dataclass(slots=True)
class SearchConfig:
    """
    Central configuration object for all search operations.
    
    Provides comprehensive settings for search scope, behavior,
    performance, and output formatting.
    """
```

### Core Configuration

#### Search Scope

```python
paths: list[str] = field(default_factory=lambda: ["."])
include: list[str] | None = None  # Auto-detect if None
exclude: list[str] | None = None  # Use defaults if None
languages: set[Language] | None = None  # Auto-detect if None
```

**Example:**

```python
config = SearchConfig(
    paths=["./src", "./tests"],
    include=["**/*.py", "**/*.pyx"],
    exclude=["**/.venv/**", "**/build/**"],
    languages={Language.PYTHON}
)
```

#### Search Behavior

```python
context: int = 2
output_format: OutputFormat = OutputFormat.TEXT
follow_symlinks: bool = False
file_size_limit: int = 2_000_000  # 2MB
```

#### Content Toggles

```python
enable_docstrings: bool = True
enable_comments: bool = True
enable_strings: bool = True
```

**Example:**

```python
# Search only in code, skip docstrings and comments
config = SearchConfig(
    enable_docstrings=False,
    enable_comments=False,
    enable_strings=True
)
```

#### Performance Settings

```python
parallel: bool = True
workers: int = 0  # 0 = auto (cpu_count)
strict_hash_check: bool = False
dir_prune_exclude: bool = True
```

**Example:**

```python
# High-performance configuration
config = SearchConfig(
    parallel=True,
    workers=8,
    strict_hash_check=False,  # Faster, less precise
    dir_prune_exclude=True    # Skip excluded directories
)
```

### Methods

#### `get_include_patterns()`

Get resolved include patterns based on language detection.

**Returns:** `list[str]` - List of glob patterns

#### `get_exclude_patterns()`

Get resolved exclude patterns with sensible defaults.

**Returns:** `list[str]` - List of glob patterns

#### `resolve_cache_dir()`

Get the resolved cache directory path.

**Returns:** `Path` - Cache directory path

**Example:**

```python
config = SearchConfig(paths=["./src"])
cache_dir = config.resolve_cache_dir()
print(f"Cache directory: {cache_dir}")
```

---

## Query and Result Types

### Query Class

Complete search query specification.

```python
@dataclass(slots=True)
class Query:
    """
    Complete search query specification with all parameters.
    """
    pattern: str
    use_regex: bool = False
    use_ast: bool = False
    use_semantic: bool = False
    context: int = 2
    output: OutputFormat = OutputFormat.TEXT
    filters: ASTFilters | None = None
    metadata_filters: MetadataFilters | None = None
    search_docstrings: bool = True
    search_comments: bool = True
    search_strings: bool = True
```

**Example:**

```python
from pysearch.types import Query, ASTFilters, OutputFormat

# Complex query example
query = Query(
    pattern="async def.*handler",
    use_regex=True,
    use_ast=True,
    context=5,
    output=OutputFormat.JSON,
    ast_filters=ASTFilters(
        func_name=".*handler",
        decorator="lru_cache"
    ),
    search_docstrings=False
)
```

### SearchResult Class

Complete search results with metadata and statistics.

```python
@dataclass(slots=True)
class SearchResult:
    """
    Complete search results with items and metadata.
    """
    items: list[SearchItem]
    stats: SearchStats
    query: Query
    errors: list[str] = field(default_factory=list)
```

**Properties:**

- `items`: List of individual search results
- `stats`: Performance and match statistics
- `query`: Original query that produced these results
- `errors`: Any errors encountered during search

### SearchItem Class

Individual search result with context and match information.

```python
@dataclass(slots=True)
class SearchItem:
    """
    Individual search result item with context.
    """
    file: Path
    start_line: int
    end_line: int
    lines: list[str]
    match_spans: list[MatchSpan] = field(default_factory=list)
    score: float = 0.0
    metadata: FileMetadata | None = None
```

**Example:**

```python
# Process search results
for item in results.items:
    print(f"Found in {item.file} (score: {item.score:.2f})")
    print(f"Lines {item.start_line}-{item.end_line}:")
    
    for i, line in enumerate(item.lines):
        line_num = item.start_line + i
        print(f"  {line_num:4d}: {line}")
    
    # Highlight matches
    for span in item.match_spans:
        line_idx, (start_col, end_col) = span
        print(f"    Match at line {item.start_line + line_idx}, cols {start_col}-{end_col}")
```

### SearchStats Class

Performance and match statistics.

```python
@dataclass(slots=True)
class SearchStats:
    """
    Search performance and match statistics.
    """
    files_scanned: int = 0
    files_matched: int = 0
    total_matches: int = 0
    elapsed_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    index_size: int = 0
```

**Example:**

```python
stats = results.stats
print(f"Scanned {stats.files_scanned} files in {stats.elapsed_ms:.1f}ms")
print(f"Found {stats.total_matches} matches in {stats.files_matched} files")
print(f"Cache efficiency: {stats.cache_hits}/{stats.cache_hits + stats.cache_misses}")
```

---

## Filters and Metadata

### ASTFilters Class

AST-based filtering for structural search.

```python
@dataclass(slots=True)
class ASTFilters:
    """
    AST-based filters for structural code search.
    """
    func_name: str | None = None
    class_name: str | None = None
    decorator: str | None = None
    imported: str | None = None
```

**Parameters:**

- `func_name` (str, optional): Regex pattern for function names
- `class_name` (str, optional): Regex pattern for class names
- `decorator` (str, optional): Regex pattern for decorator names
- `imported` (str, optional): Regex pattern for import symbols

**Example:**

```python
from pysearch.types import ASTFilters

# Find async handler functions with caching
filters = ASTFilters(
    func_name=".*handler$",
    decorator="(lru_cache|cache)",
)

# Find test classes
filters = ASTFilters(
    class_name="Test.*|.*Test$"
)

# Find specific imports
filters = ASTFilters(
    imported="requests\\.(get|post)"
)
```

### MetadataFilters Class

Advanced metadata-based filtering.

```python
@dataclass(slots=True)
class MetadataFilters:
    """
    Advanced metadata-based filters for file selection.
    """
    min_lines: int | None = None
    max_lines: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    languages: set[Language] | None = None
    authors: set[str] | None = None
    modified_after: str | None = None
    modified_before: str | None = None
```

**Example:**

```python
from pysearch.types import MetadataFilters, Language
from datetime import datetime, timedelta

# Find substantial Python files modified recently
filters = MetadataFilters(
    min_lines=100,
    max_size=1024*1024,  # 1MB
    languages={Language.PYTHON},
    modified_after="2024-01-01"
)

# Find files by specific authors
filters = MetadataFilters(
    authors={"alice", "bob"},
    min_lines=50
)
```

### FileMetadata Class

Extended file metadata for advanced operations.

```python
@dataclass(slots=True)
class FileMetadata:
    """
    Extended file metadata for advanced filtering.
    """
    path: Path
    size: int
    mtime: float
    language: Language
    encoding: str = "utf-8"
    line_count: int | None = None
    author: str | None = None
    created_date: float | None = None
    modified_date: float | None = None
```

---

## Utility Functions

### Language Detection

#### `detect_language(file_path)`

Detect programming language from file path/extension.

**Parameters:**

- `file_path` (Path | str): File path to analyze

**Returns:** `Language` - Detected language or Language.UNKNOWN

**Example:**

```python
from pysearch.language_detection import detect_language

lang = detect_language("example.py")
print(f"Detected language: {lang}")  # Language.PYTHON
```

#### `get_supported_languages()`

Get list of all supported programming languages.

**Returns:** `list[Language]` - List of supported languages

**Example:**

```python
from pysearch.language_detection import get_supported_languages

languages = get_supported_languages()
print(f"Supported: {[lang.value for lang in languages]}")
```

### Metadata Utilities

#### `create_metadata_filters(**kwargs)`

Create MetadataFilters with validation.

**Parameters:**

- `**kwargs`: Filter parameters

**Returns:** `MetadataFilters` - Validated filter object

**Example:**

```python
from pysearch.utils import create_metadata_filters

filters = create_metadata_filters(
    min_lines=50,
    languages=["python", "javascript"],
    modified_after="2024-01-01"
)
```

#### `create_file_metadata(file_path)`

Create FileMetadata object for a file.

**Parameters:**

- `file_path` (Path): File to analyze

**Returns:** `FileMetadata` - File metadata object

### Logging Configuration

#### `configure_logging(level="INFO", format=None)`

Configure pysearch logging.

**Parameters:**

- `level` (str): Log level (DEBUG, INFO, WARNING, ERROR)
- `format` (str, optional): Custom log format

**Example:**

```python
from pysearch.logging_config import configure_logging

# Enable debug logging
configure_logging(level="DEBUG")

# Custom format
configure_logging(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

#### `enable_debug_logging()`

Enable debug logging with detailed output.

#### `disable_logging()`

Disable all pysearch logging.

---

## Exception Classes

### SearchError

Base exception for all search-related errors.

```python
class SearchError(Exception):
    """Base exception for search operations."""
```

### FileAccessError

Raised when file access fails.

```python
class FileAccessError(SearchError):
    """Raised when file cannot be accessed."""
```

### PermissionError

Raised when insufficient permissions.

```python
class PermissionError(SearchError):
    """Raised when permission denied."""
```

### EncodingError

Raised when file encoding issues occur.

```python
class EncodingError(SearchError):
    """Raised when file encoding cannot be determined."""
```

### ParsingError

Raised when AST parsing fails.

```python
class ParsingError(SearchError):
    """Raised when AST parsing fails."""
```

**Example Error Handling:**

```python
from pysearch import PySearch, SearchError, FileAccessError

try:
    engine = PySearch()
    results = engine.search("pattern")
except FileAccessError as e:
    print(f"File access error: {e}")
except ParsingError as e:
    print(f"Parsing error: {e}")
except SearchError as e:
    print(f"Search error: {e}")
```

---

## Enumerations

### OutputFormat

Available output formats.

```python
class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    HIGHLIGHT = "highlight"
```

### Language

Supported programming languages.

```python
class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    RUST = "rust"
    GO = "go"
    # ... and more
```

---

## Advanced Usage Patterns

### Batch Processing

```python
from pysearch import PySearch, SearchConfig
from pysearch.types import Query

# Setup for batch processing
config = SearchConfig(
    paths=["./src"],
    parallel=True,
    workers=8
)
engine = PySearch(config)

# Process multiple queries
queries = [
    Query(pattern="def.*handler", use_regex=True),
    Query(pattern="class.*Test", use_regex=True),
    Query(pattern="import requests", use_ast=True)
]

results = []
for query in queries:
    result = engine.run(query)
    results.append(result)
    print(f"Query '{query.pattern}': {len(result.items)} matches")
```

### Custom Result Processing

```python
def process_results(results):
    """Custom result processing with grouping and filtering."""

    # Group by file
    by_file = {}
    for item in results.items:
        if item.file not in by_file:
            by_file[item.file] = []
        by_file[item.file].append(item)

    # Sort by score
    for file_path, items in by_file.items():
        items.sort(key=lambda x: x.score, reverse=True)

        print(f"\n{file_path}:")
        for item in items[:3]:  # Top 3 matches per file
            print(f"  Score: {item.score:.2f}")
            print(f"  Lines {item.start_line}-{item.end_line}")
```

### Integration with External Tools

```python
import json
from pathlib import Path

def export_to_json(results, output_file):
    """Export results to JSON for external processing."""

    data = {
        "query": {
            "pattern": results.query.pattern,
            "use_regex": results.query.use_regex,
            "use_ast": results.query.use_ast
        },
        "stats": {
            "files_scanned": results.stats.files_scanned,
            "total_matches": results.stats.total_matches,
            "elapsed_ms": results.stats.elapsed_ms
        },
        "matches": [
            {
                "file": str(item.file),
                "start_line": item.start_line,
                "end_line": item.end_line,
                "score": item.score,
                "lines": item.lines
            }
            for item in results.items
        ]
    }

    Path(output_file).write_text(json.dumps(data, indent=2))
```

---

## GraphRAG API

GraphRAG (Graph Retrieval-Augmented Generation) combines knowledge graphs with vector similarity for intelligent code search.

### GraphRAGEngine

```python
from pysearch.analysis.graphrag import GraphRAGEngine, KnowledgeGraphBuilder

# Build knowledge graph
builder = KnowledgeGraphBuilder()
graph = builder.build_from_codebase("./src")

# Create engine and search
engine = GraphRAGEngine(graph)
results = engine.search(
    query="authentication handler",
    entity_types=["function", "class"],
    max_depth=3
)
```

### EntityExtractor

```python
from pysearch.analysis.graphrag import EntityExtractor

extractor = EntityExtractor()
entities = extractor.extract_from_file("src/auth.py")
for entity in entities:
    print(f"{entity.type}: {entity.name} (line {entity.start_line})")
```

### RelationshipMapper

```python
from pysearch.analysis.graphrag import RelationshipMapper

mapper = RelationshipMapper()
relationships = mapper.map_relationships(entities)
for rel in relationships:
    print(f"{rel.source} --[{rel.type}]--> {rel.target}")
```

### GraphRAG Types

```python
from pysearch.core.types import (
    GraphRAGQuery,      # Query specification for GraphRAG
    GraphRAGResult,     # Result from GraphRAG search
    KnowledgeGraph,     # Knowledge graph container
    CodeEntity,         # Extracted code entity
    EntityRelationship, # Relationship between entities
    EntityType,         # Entity type enum (FUNCTION, CLASS, etc.)
    RelationType,       # Relationship type enum (CALLS, INHERITS, etc.)
)
```

---

## Fuzzy Search API

Fuzzy search supports typo-tolerant matching with multiple algorithms.

### Fuzzy Search Exports

```python
from pysearch.search import (
    fuzzy_search_advanced,  # Advanced fuzzy search
    fuzzy_match,            # Single fuzzy match
    calculate_similarity,   # Similarity score
    suggest_corrections,    # Spelling corrections
    fuzzy_regex_pattern,    # Generate fuzzy regex
    FuzzyAlgorithm,         # Algorithm enum
    FuzzyMatch,             # Match result
)
```

### Fuzzy Search Example

```python
from pysearch.search import fuzzy_search_advanced, FuzzyAlgorithm

# Search with typo tolerance
results = fuzzy_search_advanced(
    query="authetication",  # typo
    candidates=["authentication", "authorization", "audit"],
    algorithm=FuzzyAlgorithm.LEVENSHTEIN,
    threshold=0.7
)

for match in results:
    print(f"{match.text} (score: {match.score:.2f})")
```

### Algorithms

- `FuzzyAlgorithm.LEVENSHTEIN`: Edit distance-based similarity
- `FuzzyAlgorithm.JARO_WINKLER`: Jaro-Winkler distance
- `FuzzyAlgorithm.NGRAM`: N-gram based similarity

---

## Boolean Search API

Boolean search supports logical query composition with `AND`, `OR`, `NOT` operators.

### Boolean Search Exports

```python
from pysearch.search import (
    parse_boolean_query,                # Parse query string
    evaluate_boolean_query,             # Evaluate against text
    evaluate_boolean_query_with_items,  # Evaluate against items
    extract_terms,                      # Extract search terms
    BooleanQueryParser,                 # Parser class
    BooleanQueryEvaluator,              # Evaluator class
)
```

### Boolean Search Example

```python
from pysearch.search import parse_boolean_query, evaluate_boolean_query

# Parse a boolean query
query = parse_boolean_query("(async AND handler) NOT test")

# Evaluate against text
matches = evaluate_boolean_query(query, "async def request_handler():")
print(f"Matches: {matches}")  # True
```

---

## Dependency Analysis API

Analyze import dependencies, detect circular imports, and calculate coupling metrics.

### DependencyAnalyzer

```python
from pysearch.analysis import DependencyAnalyzer, DependencyGraph, DependencyMetrics

analyzer = DependencyAnalyzer()
graph = analyzer.analyze("./src")

# Get metrics
metrics = analyzer.calculate_metrics(graph)
print(f"Total modules: {metrics.total_modules}")
print(f"Total edges: {metrics.total_edges}")
print(f"Avg coupling: {metrics.avg_coupling:.2f}")
```

### CircularDependencyDetector

```python
from pysearch.analysis import CircularDependencyDetector

detector = CircularDependencyDetector(graph)
cycles = detector.find_cycles()
for cycle in cycles:
    print(f"Circular: {' -> '.join(cycle)}")
```

---

## Multi-Repository API

Search across multiple repositories with priority-based result aggregation.

### MultiRepoSearchEngine

```python
from pysearch.integrations import MultiRepoSearchEngine, RepositoryInfo

engine = MultiRepoSearchEngine()

# Add repositories with priorities
engine.add_repository("main", "./main-app", priority=1.0)
engine.add_repository("lib", "./shared-lib", priority=0.8)
engine.add_repository("tests", "./test-suite", priority=0.5)

# Search across all repositories
results = engine.search_all("async def", regex=True)

for repo_name, repo_results in results.repository_results.items():
    print(f"{repo_name}: {len(repo_results.items)} matches")
```

---

## Storage API

Vector database abstraction layer with multiple backend support and embedding providers.

### VectorDatabase

```python
from pysearch.storage import (
    VectorDatabase,          # Abstract base
    LanceDBProvider,         # LanceDB backend
    QdrantProvider,          # Qdrant backend
    ChromaProvider,          # Chroma backend
    VectorIndexManager,      # Index management
    VectorSearchResult,      # Search result
)
```

### Embedding Providers

```python
from pysearch.storage import (
    EmbeddingConfig,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
)

# Configure OpenAI embeddings
config = EmbeddingConfig(
    provider="openai",
    model="text-embedding-ada-002",
    api_key="your-api-key"
)
provider = OpenAIEmbeddingProvider(config)
```

### QdrantVectorStore

```python
from pysearch.storage import QdrantVectorStore, QdrantConfig

config = QdrantConfig(host="localhost", port=6333)
store = QdrantVectorStore(config)

# Search by vector similarity
results = store.search(
    collection="code_embeddings",
    query_vector=embedding,
    limit=10
)
```

---

## IDE Integration API

LSP-like features for IDE integration including jump-to-definition, find references, and more.

### IDEHooks

```python
from pysearch.integrations import (
    IDEHooks,
    IDEIntegration,
    DefinitionLocation,
    ReferenceLocation,
    CompletionItem,
    HoverInfo,
    DocumentSymbol,
    Diagnostic,
    ide_query,
)

hooks = IDEHooks(engine)

# Jump to definition
definition = hooks.go_to_definition("MyClass", "src/main.py", line=42)
print(f"Defined at: {definition.file}:{definition.line}")

# Find all references
refs = hooks.find_references("MyClass")
for ref in refs:
    print(f"  {ref.file}:{ref.line}")

# Get completions
completions = hooks.get_completions("src/main.py", line=10, col=5)
for item in completions:
    print(f"  {item.label} ({item.kind})")

# Get hover info
info = hooks.hover("src/main.py", line=42, col=10)
print(f"  {info.contents}")

# List document symbols
symbols = hooks.document_symbols("src/main.py")
for sym in symbols:
    print(f"  {sym.kind}: {sym.name} (line {sym.range.start})")
```

---

## Performance Considerations

### Optimization Tips

1. **Use appropriate include/exclude patterns** to limit search scope
2. **Enable parallel processing** for large codebases
3. **Configure caching** for repeated searches
4. **Use AST filters** to narrow structural searches
5. **Set reasonable context limits** to avoid excessive output
6. **Disable unnecessary content types** (docstrings, comments, strings)

### Memory Management

```python
# For very large codebases
config = SearchConfig(
    file_size_limit=1_000_000,  # 1MB limit
    workers=4,  # Limit parallel workers
    strict_hash_check=False  # Reduce I/O
)
```

### Monitoring Performance

```python
results = engine.search("pattern")
stats = results.stats

print(f"Performance metrics:")
print(f"  Files scanned: {stats.files_scanned}")
print(f"  Elapsed time: {stats.elapsed_ms:.1f}ms")
print(f"  Cache efficiency: {stats.cache_hits / (stats.cache_hits + stats.cache_misses):.2%}")
```
