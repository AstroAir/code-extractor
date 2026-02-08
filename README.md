# pysearch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

pysearch is a high-performance, context-aware search engine for Python codebases that supports text/regex/AST/semantic/fuzzy/boolean/GraphRAG search, providing CLI, Python API, and MCP (Model Context Protocol) interfaces, designed for engineering-grade retrieval in large multi-module projects.

## Features Overview

### Search Capabilities

- **Text/Regex Search**: Enhanced regex via `regex` library with multiline mode and named groups
- **AST Structural Search**: Python AST-based matching with function/class/decorator/import filters
- **Semantic Search**: Lightweight symbolic + optional transformer-based embedding search
- **Fuzzy Search**: Multiple algorithms (Levenshtein, Jaro-Winkler, n-gram) with typo tolerance via `rapidfuzz`
- **Boolean Search**: Logical operators (`AND`, `OR`, `NOT`) for complex query composition
- **GraphRAG Search**: Graph Retrieval-Augmented Generation combining knowledge graphs with vector similarity

### Indexing & Storage

- **Multi-Index Architecture**: Code snippets, full-text (SQLite FTS5), chunk, and vector indexes
- **Content-Addressed Caching**: SHA256-based deduplication with tag-based index management
- **Vector Database Support**: LanceDB, Qdrant, and Chroma backends
- **Embedding Providers**: OpenAI, HuggingFace, and local model support
- **Distributed Indexing**: Multi-process parallel indexing for large codebases
- **Advanced Chunking**: Code-aware chunking with tree-sitter integration for 20+ languages

### Analysis & Intelligence

- **Dependency Analysis**: Import graph generation, circular dependency detection, and metrics
- **GraphRAG Engine**: Entity extraction, relationship mapping, and knowledge graph construction
- **Language Detection**: Automatic programming language identification for 20+ languages
- **Content Addressing**: SHA256-based content deduplication across branches

### Developer Experience

- **CLI & Python API**: Full-featured command-line interface and programmable API
- **MCP Integration**: Model Context Protocol servers for LLM tool integration
- **IDE Integration**: Jump-to-definition, find references, completions, hover info, and diagnostics
- **File Watcher**: Real-time file monitoring with automatic index updates
- **Search History**: Session tracking, bookmarks, and search analytics
- **Multiple Output Formats**: Plain text, JSON, syntax-highlighted console output
- **Performance Monitoring**: Real-time profiling, metrics collection, and optimization suggestions

### Reliability & Performance

- **Parallel Processing**: Multi-threaded file processing with configurable worker pools
- **Advanced Error Handling**: Circuit breaker pattern, recovery manager, and error aggregation
- **Configurable Scoring**: Pluggable ranking strategies with result deduplication and clustering
- **Smart Caching**: Multi-level cache (in-memory + disk) with LRU eviction and TTL

## Installation

Requires Python 3.10+.

### Basic Installation

```bash
pip install -e .
```

### Optional Dependencies

```bash
# File watching support
pip install -e ".[watch]"

# GraphRAG and enhanced analysis
pip install -e ".[graphrag]"

# Vector database support
pip install -e ".[vector]"

# Advanced semantic search (transformer models)
pip install -e ".[semantic]"

# All optional features
pip install -e ".[all]"
```

### Development Setup

```bash
# Using the setup script
./scripts/dev-install.sh

# Or manually
pip install -e ".[dev]"
pre-commit install
```

### Validation

```bash
make validate
# or
./scripts/validate-project.sh
```

## Quick Start

### CLI Usage

Find all function definitions:
```bash
pysearch find --pattern "def " --path ./src --include "**/*.py"
```

Regex search with context:
```bash
pysearch find \
  --pattern "def.*handler" \
  --regex \
  --context 3 \
  --format highlight
```

AST-based search with filters:
```bash
pysearch find \
  --pattern "def" \
  --ast \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache"
```

Semantic search:
```bash
pysearch find \
  --pattern "database connection" \
  --semantic \
  --context 5
```

Boolean queries:
```bash
pysearch find \
  --pattern "(async AND handler) NOT test" \
  --logic \
  --context 3
```

Count-only (fast):
```bash
pysearch find --pattern "def" --count
```

### Python API

Basic search:
```python
from pysearch import PySearch, SearchConfig

config = SearchConfig(
    paths=["."],
    include=["**/*.py"],
    context=3
)

engine = PySearch(config)
results = engine.search("def main")
print(f"Found {len(results.items)} matches in {results.stats.elapsed_ms}ms")

for item in results.items:
    print(f"\n{item.file}:{item.start_line}-{item.end_line}")
    for line in item.lines:
        print(f"  {line}")
```

Advanced query with AST filters:
```python
from pysearch.core.types import Query, ASTFilters

filters = ASTFilters(
    func_name=".*handler",
    decorator="lru_cache|cache",
    imported="requests\\.(get|post)"
)

query = Query(
    pattern="def",
    use_ast=True,
    use_regex=True,
    ast_filters=filters,
    context=5
)

results = engine.run(query)
```

Multi-repository search:
```python
from pysearch.integrations import MultiRepoSearchEngine

multi_engine = MultiRepoSearchEngine()
multi_engine.add_repository("frontend", "./frontend")
multi_engine.add_repository("backend", "./backend")

results = multi_engine.search_all("async def")
for repo_name, repo_results in results.repository_results.items():
    print(f"{repo_name}: {len(repo_results.items)} matches")
```

GraphRAG search:
```python
from pysearch import PySearch, SearchConfig, GraphRAGQuery

config = SearchConfig(paths=["./src"])
engine = PySearch(config)

# Build knowledge graph from codebase
engine.build_knowledge_graph()

# Query with GraphRAG
query = GraphRAGQuery(
    query="functions related to authentication",
    entity_types=["function", "class"],
    max_depth=3,
    use_embeddings=True
)
results = engine.graphrag_search(query)
```

Fuzzy search:
```python
from pysearch.search import fuzzy_search_advanced, FuzzyAlgorithm

results = fuzzy_search_advanced(
    query="authetication",  # typo intended
    candidates=code_symbols,
    algorithm=FuzzyAlgorithm.LEVENSHTEIN,
    threshold=0.7
)
```

## Project Structure

```text
src/pysearch/
├── core/                  # Core engine, config, types, history, managers
│   ├── api.py             # Main PySearch engine class
│   ├── config.py          # SearchConfig and RankStrategy
│   ├── types/             # Core data types (basic, GraphRAG)
│   ├── history/           # Search history, bookmarks, analytics, sessions
│   └── managers/          # Internal manager modules
│       ├── hybrid_search.py              # Semantic and hybrid search
│       ├── graphrag_integration.py       # GraphRAG knowledge graph
│       ├── ide_integration.py            # IDE hooks (go-to-def, references)
│       ├── distributed_indexing_integration.py  # Distributed parallel indexing
│       ├── multi_repo_integration.py     # Multi-repository search
│       ├── dependency_integration.py     # Dependency analysis
│       ├── file_watching.py              # Real-time file monitoring
│       ├── cache_integration.py          # Cache management
│       ├── indexing_integration.py       # Metadata indexing
│       └── parallel_processing.py        # Parallel search execution
├── search/                # Search strategies and pattern matching
│   ├── matchers.py        # Text, regex, AST matching
│   ├── boolean.py         # Boolean query parser and evaluator
│   ├── fuzzy.py           # Fuzzy search (multiple algorithms)
│   ├── scorer.py          # Ranking, deduplication, clustering
│   ├── semantic.py        # Lightweight semantic search
│   └── semantic_advanced.py  # Transformer-based semantic search
├── analysis/              # Code analysis and understanding
│   ├── graphrag/          # GraphRAG engine (entity extraction, relationships)
│   ├── dependency_analysis.py   # Dependency graphs and circular detection
│   ├── language_detection.py    # Language identification
│   ├── language_support.py      # Tree-sitter multi-language processing
│   └── content_addressing.py    # Content-addressed indexing
├── indexing/              # Indexing and caching systems
│   ├── indexer.py         # Core file indexer
│   ├── advanced/          # Advanced indexing (chunking, coordinator, engine)
│   ├── cache/             # Cache backends, cleanup, statistics
│   ├── indexes/           # Specialized indexes (full-text, vector, chunk, snippets)
│   └── metadata/          # Metadata indexer and database
├── integrations/          # External integrations
│   ├── multi_repo.py      # Multi-repository search
│   ├── distributed_indexing.py  # Distributed parallel indexing
│   └── ide_hooks.py       # IDE integration (LSP-like features)
├── storage/               # Data storage and persistence
│   ├── vector_db.py       # Vector DB abstraction (LanceDB, Qdrant, Chroma)
│   └── qdrant_client.py   # Qdrant vector store client
├── utils/                 # Utilities and helpers
│   ├── error_handling.py          # Error hierarchy and collectors
│   ├── advanced_error_handling.py # Circuit breaker, recovery manager
│   ├── file_watcher.py            # File system monitoring
│   ├── performance_monitoring.py  # Profiling and metrics
│   ├── formatter.py               # Output formatting
│   ├── helpers.py                 # Common utilities
│   ├── logging_config.py          # Logging configuration
│   └── metadata_filters.py       # Metadata-based filtering
└── cli/                   # Command-line interface
    └── main.py            # CLI entry point (Click-based)

mcp/                       # MCP (Model Context Protocol) servers
├── servers/               # MCP server implementations
├── shared/                # Shared MCP utilities
└── README.md              # MCP documentation

tests/                     # Test suite (unit, integration, performance)
docs/                      # Comprehensive documentation
scripts/                   # Build and development scripts
configs/                   # Configuration examples
```

### MCP Servers

PySearch includes MCP server implementations for LLM integration:

```bash
./scripts/run-mcp-server.sh main
```

Documentation:
- [MCP Overview](docs/mcp/index.md) - Introduction to MCP integration
- [MCP Tutorial](docs/mcp/tutorial.md) - Step-by-step setup guide
- [MCP API Reference](docs/mcp/api.md) - All MCP tool documentation
- [Advanced MCP Features](docs/mcp/advanced.md) - Advanced capabilities

## Core Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                       │
├─────────────────────────────────────────────────────────────────┤
│  CLI (Click)  │  Python API  │  MCP Servers  │  IDE Hooks      │
├─────────────────────────────────────────────────────────────────┤
│                       Core Engine Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  PySearch  │  Query Engine  │  GraphRAG  │  Search History      │
├─────────────────────────────────────────────────────────────────┤
│                     Processing Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  Matchers  │  Fuzzy  │  Boolean  │  Semantic  │  Scorer         │
├─────────────────────────────────────────────────────────────────┤
│                    Indexing & Storage Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Indexer  │  Cache  │  Vector DB  │  Metadata  │  Chunks        │
├─────────────────────────────────────────────────────────────────┤
│                      Foundation Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Language Detection  │  File Watcher  │  Error Handling  │ Utils│
└─────────────────────────────────────────────────────────────────┘
```

## Common Use Cases

### Code Navigation and Discovery

```bash
# Find all function definitions
pysearch find --pattern "def " --path ./src --include "**/*.py"

# Locate all class definitions
pysearch find --pattern "class " --path . --regex --filter-class-name ".*"

# Find imports of specific modules
pysearch find --pattern "from requests import" --path . --context 2
```

### Refactoring and Code Analysis

```bash
# Find all usages of a specific function
pysearch find --pattern "deprecated_function" --path . --context 3

# Locate all TODO comments
pysearch find --pattern "TODO|FIXME|HACK" --regex --comments

# Find functions with specific decorators
pysearch find --pattern "def" --ast --filter-decorator "lru_cache|cache"
```

### Code Quality and Security

```bash
# Search for potential security issues
pysearch find --pattern "eval|exec|subprocess" --regex --context 5

# Find async/await patterns
pysearch find --pattern "async def|await " --regex --semantic

# Locate error handling patterns
pysearch find --pattern "try:|except:|finally:" --regex --context 3
```

### Dependency Analysis

```python
from pysearch.analysis import DependencyAnalyzer, CircularDependencyDetector

analyzer = DependencyAnalyzer("./src")
graph = analyzer.build_dependency_graph()

# Detect circular dependencies
detector = CircularDependencyDetector(graph)
cycles = detector.find_cycles()
for cycle in cycles:
    print(f"Circular dependency: {' -> '.join(cycle)}")

# Get dependency metrics
metrics = analyzer.calculate_metrics()
print(f"Total modules: {metrics.total_modules}")
print(f"Avg coupling: {metrics.avg_coupling:.2f}")
```

## CLI Reference

```bash
pysearch find \
  --path src tests \
  --include "**/*.py" \
  --exclude "**/.venv/**" "**/build/**" \
  --pattern "def .*_handler" \
  --regex \
  --context 4 \
  --format json \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache" \
  --rank "ast_weight:2,text_weight:1"
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--path` | Search paths (multiple allowed) |
| `--include/--exclude` | Include/exclude glob patterns |
| `--pattern` | Text/regex pattern or semantic query |
| `--regex` | Enable regex matching |
| `--ast` | Enable AST structural search |
| `--semantic` | Enable semantic search |
| `--logic` | Enable boolean query mode |
| `--context` | Number of context lines |
| `--format` | Output format (`text`/`json`/`highlight`) |
| `--count` | Count-only mode (fast) |
| `--max-per-file` | Limit results per file |
| `--filter-func-name` | Filter by function name (regex) |
| `--filter-class-name` | Filter by class name (regex) |
| `--filter-decorator` | Filter by decorator (regex) |
| `--filter-import` | Filter by import (regex) |
| `--parallel/--no-parallel` | Toggle parallel processing |
| `--workers` | Number of worker threads |
| `--cache/--no-cache` | Toggle result caching |
| `--stats` | Print performance statistics |

See [CLI Reference](docs/guide/cli-reference.md) for complete documentation.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pysearch

# Run specific test categories
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m benchmark      # Performance benchmarks
pytest -m "not slow"     # Skip slow tests
```

## Documentation

- [Installation Guide](docs/getting-started/installation.md) - System requirements and setup
- [Usage Guide](docs/guide/usage.md) - Getting started and common workflows
- [CLI Reference](docs/guide/cli-reference.md) - Complete command-line reference
- [API Reference](docs/api/index.md) - Python API documentation
- [Configuration Guide](docs/guide/configuration.md) - All configuration options
- [Architecture](docs/advanced/architecture.md) - System design and internals
- [Advanced Features](docs/advanced/features.md) - Enhanced indexing engine
- [GraphRAG Guide](docs/advanced/graphrag.md) - Knowledge graph search
- [Advanced Indexing](docs/advanced/indexing-guide.md) - Indexing engine deep dive
- [Performance Tuning](docs/guide/performance.md) - Optimization guide
- [MCP Integration](docs/mcp/index.md) - Model Context Protocol
- [Troubleshooting](docs/help/troubleshooting.md) - Common issues and solutions
- [FAQ](docs/help/faq.md) - Frequently asked questions

## License

MIT