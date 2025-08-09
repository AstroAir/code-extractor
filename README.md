# pysearch

pysearch is a high-performance, context-aware search engine for Python codebases that supports text/regex/AST/semantic search, providing both CLI and programmable API interfaces, designed for engineering-grade retrieval in large multi-module projects.

## Features Overview

- **Code Block Matching**: Functions, classes, decorators, imports, strings/comments, arbitrary code snippets
- **Context-Aware**: Returns matched code with configurable context lines
- **Project-Wide Search**: Efficient indexing and caching, optimized for large codebases
- **Multiple Match Types**: Regex, AST structural, semantic (lightweight vector/symbolic features)
- **Highly Customizable**: Include/exclude directories, file types, context windows, filters (function names, class names, decorators, imports, etc.)
- **Multiple Output Formats**: Plain text, JSON, highlighted console output
- **Scoring and Ranking**: Configurable result scoring rules
- **Performance Metrics**: Search time, files scanned, match counts, etc.
- **CLI & API**: Command-line operations and Python embedded calls
- **Testing & Benchmarks**: pytest coverage > 90%, with benchmark scripts

## Installation

Recommended Python 3.10+.

### Basic Installation

```bash
pip install -e .
```

### Development Setup

For development, use the provided setup script:

```bash
./scripts/dev-install.sh
```

Or manually:

```bash
pip install -e ".[dev]"
pre-commit install
```

### Validation

Verify your installation:

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

Search for handler functions with regex:
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

Semantic search for database-related code:
```bash
pysearch find \
  --pattern "database connection" \
  --semantic \
  --context 5
```

### API Usage

Basic search:
```python
from pysearch.api import PySearch
from pysearch.config import SearchConfig

# Create search configuration
config = SearchConfig(
    paths=["."],
    include=["**/*.py"],
    context=3
)

# Initialize search engine
engine = PySearch(config)

# Perform search
results = engine.search("def main")
print(f"Found {len(results.items)} matches in {results.stats.elapsed_ms}ms")

# Process results
for item in results.items:
    print(f"\n{item.file}:{item.start_line}-{item.end_line}")
    for line in item.lines:
        print(f"  {line}")
```

Advanced query with filters:
```python
from pysearch.types import Query, ASTFilters

# Create AST filters
filters = ASTFilters(
    func_name=".*handler",
    decorator="lru_cache|cache",
    imported="requests\\.(get|post)"
)

# Create advanced query
query = Query(
    pattern="def",
    use_ast=True,
    use_regex=True,
    ast_filters=filters,
    context=5
)

# Execute query
results = engine.run(query)
```

Multi-repository search:
```python
from pysearch.multi_repo import MultiRepoSearchEngine

# Initialize multi-repo engine
multi_engine = MultiRepoSearchEngine()

# Add repositories
multi_engine.add_repository("frontend", "./frontend")
multi_engine.add_repository("backend", "./backend")
multi_engine.add_repository("shared", "./shared-lib")

# Search across all repositories
results = multi_engine.search_all("async def")

# Process results by repository
for repo_name, repo_results in results.repository_results.items():
    print(f"{repo_name}: {len(repo_results.items)} matches")
```

## Core Capabilities

- **Text/Regex Search**: Enhanced regex capabilities based on `regex` library, supporting multiline mode and named groups
- **AST Search**: Based on `ast` module and custom matchers, filter or locate nodes by function/class/decorator/import
- **Semantic Search**: Lightweight vector + symbolic features, considering structure and identifier semantics (no external models required)
- **Indexing & Caching**: Records file mtime, hash, size for incremental updates
- **Output & Highlighting**: `rich`/`pygments` console highlighting, `orjson` fast JSON output

## Project Structure

```text
├── src/pysearch/          # Core PySearch library
├── mcp/                   # MCP (Model Context Protocol) servers
│   ├── servers/           # MCP server implementations
│   ├── shared/            # Shared MCP utilities
│   └── README.md          # MCP documentation
├── tools/                 # Development and utility tools
├── examples/              # Usage examples and demos
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Build and development scripts
└── configs/               # Configuration files
```

### MCP Servers

PySearch includes several MCP server implementations for LLM integration:

- **Main MCP Server**: Advanced features with fuzzy search, analysis, and composition
- **Basic MCP Server**: Core search functionality (legacy)
- **FastMCP Server**: Optimized performance implementation

Run an MCP server:

```bash
./scripts/run-mcp-server.sh main
```

#### MCP Documentation

Comprehensive documentation for MCP integration is available:
- [MCP Overview](docs/mcp-overview.md) - Introduction to MCP integration
- [MCP Tutorial](docs/mcp-tutorial.md) - Step-by-step guide to using MCP with PySearch
- [MCP API Reference](docs/mcp-api.md) - Detailed API documentation for all MCP tools
- [Advanced MCP Features](docs/mcp-advanced.md) - In-depth coverage of advanced capabilities

## Common Use Cases

### Code Navigation and Discovery

Find all function definitions in a project:
```bash
pysearch find --pattern "def " --path ./src --include "**/*.py"
```

Locate all class definitions:
```bash
pysearch find --pattern "class " --path . --regex --filter-class-name ".*"
```

Find imports of specific modules:
```bash
pysearch find --pattern "from requests import" --path . --context 2
```

### Refactoring and Code Analysis

Find all usages of a specific function:
```bash
pysearch find --pattern "deprecated_function" --path . --context 3
```

Locate all TODO comments:
```bash
pysearch find --pattern "TODO|FIXME|HACK" --regex --comments
```

Find functions with specific decorators:
```bash
pysearch find --pattern "def" --ast --filter-decorator "lru_cache|cache"
```

### Code Quality and Patterns

Search for potential security issues:
```bash
pysearch find --pattern "eval|exec|subprocess" --regex --context 5
```

Find async/await patterns:
```bash
pysearch find --pattern "async def|await " --regex --semantic
```

Locate error handling patterns:
```bash
pysearch find --pattern "try:|except:|finally:" --regex --context 3
```

### Documentation and Learning

Find examples of specific patterns:
```bash
pysearch find --pattern "database connection" --semantic --context 10
```

Locate test functions:
```bash
pysearch find --pattern "test_" --path ./tests --filter-func-name "test_.*"
```

Search for configuration patterns:
```bash
pysearch find --pattern "config|settings" --semantic --include "**/*.py"
```

## CLI Usage

```bash
pysearch find \
  --path src tests \
  --include "**/*.py" \
  --exclude "*/.venv/*" "*/build/*" \
  --pattern "def .*_handler" \
  --regex \
  --context 4 \
  --format json \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache" \
  --rank "ast_weight:2,text_weight:1"
```

### Main Parameters

- `--path`: Search paths (multiple allowed)
- `--include/--exclude`: Include/exclude glob patterns
- `--pattern`: Text/regex pattern or semantic query
- `--regex`: Enable regex matching
- `--context`: Number of context lines
- `--format`: Output format (text/json/highlight)
- `--filter-func-name/--filter-class-name/--filter-decorator/--filter-import`: AST filters
- `--rank`: Ranking weight configuration
- `--docstrings/--comments/--strings`: Whether to search docstrings, comments, string literals
- `--stats`: Print performance statistics

## Programming Interface

```python
from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Query, OutputFormat

cfg = SearchConfig(
    paths=["."],
    include=["**/*.py"],
    exclude=["**/.venv/**", "**/build/**"],
    context=3,
    output_format=OutputFormat.JSON,
    enable_docstrings=True,
    enable_comments=True,
    enable_strings=True,
)

engine = PySearch(cfg)
res = engine.run(Query(pattern="ClassName", use_regex=False, use_ast=True))
print(res.stats, len(res.items))
```

## Testing & Benchmarks

Run tests with coverage:

```bash
pytest
```

Run benchmarks:

```bash
pytest tests/benchmarks -k benchmark -q
```

## Roadmap

- **Enhanced Semantic Search**: Optional external embedding backends
- **IDE/Editor Integration**: VS Code/JetBrains with protocol-based output
- **Parallel & Distributed Indexing**: Multi-process/multi-threaded indexing
- **Advanced Syntax Highlighting**: More refined highlighting and differentiated display

## License

MIT