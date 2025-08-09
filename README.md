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
```bash
pysearch find \
  --pattern "requests.get" \
  --path . \
  --regex \
  --context 3 \
  --format text
```

### API Usage
```python
from pysearch.api import PySearch
from pysearch.config import SearchConfig

engine = PySearch(SearchConfig(paths=["."], include=["**/*.py"], context=2))
results = engine.search(pattern="def main", regex=True)
for r in results.items:
    print(r.file, r.lines)
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

## Typical Use Cases

- Find all functions using specific decorators
- Locate files importing specific modules with context
- Search all code blocks containing certain regex patterns
- Find all class/function definitions named X using AST
- Cross-project statistics of match results and performance metrics

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