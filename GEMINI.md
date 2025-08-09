# PySearch Project Context

## Project Overview

PySearch is a high-performance, context-aware search engine for Python codebases that supports multiple search modes including text/regex, AST (Abstract Syntax Tree), and semantic search. It provides both CLI and programmable API interfaces, designed for engineering-grade retrieval in large multi-module projects.

### Core Features

- **Code Block Matching**: Functions, classes, decorators, imports, strings/comments, arbitrary code snippets
- **Context-Aware**: Returns matched code with configurable context lines
- **Project-Wide Search**: Efficient indexing and caching, optimized for large codebases
- **Multiple Match Types**: Regex, AST structural, semantic (lightweight vector/symbolic features)
- **Highly Customizable**: Include/exclude directories, file types, context windows, filters
- **Multiple Output Formats**: Plain text, JSON, highlighted console output
- **Scoring and Ranking**: Configurable result scoring rules
- **Performance Metrics**: Search time, files scanned, match counts, etc.
- **CLI & API**: Command-line operations and Python embedded calls
- **Testing & Benchmarks**: pytest coverage > 90%, with benchmark scripts

### Technology Stack

- **Language**: Python 3.10+
- **Dependencies**: 
  - `rich` for console output and highlighting
  - `orjson` for fast JSON serialization
  - `regex` for enhanced regex capabilities
  - `click` for CLI interface
  - `pydantic` for data validation
  - `fastmcp` for Model Context Protocol integration
  - `rapidfuzz` for fuzzy matching
- **Development Tools**:
  - `pytest` for testing
  - `mypy` for type checking
  - `ruff` and `black` for linting and formatting
  - `mkdocs` for documentation

## Project Structure

```
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

### Core Modules

- `api.py`: Main search engine class that orchestrates all search operations
- `cli.py`: Command-line interface using Click
- `config.py`: Configuration management with SearchConfig dataclass
- `types.py`: Core data types and enumerations
- `matchers.py`: Text/regex/AST matching logic
- `indexer.py`: File indexing and caching
- `scorer.py`: Result scoring and ranking
- `semantic.py`: Lightweight semantic search
- `cache_manager.py`: File content caching with TTL
- `file_watcher.py`: File change monitoring

## Building and Running

### Installation

```bash
# Basic installation
pip install -e .

# Development setup
pip install -e ".[dev]"
pre-commit install
```

Or use the provided script:
```bash
./scripts/dev-install.sh
```

### CLI Usage

```bash
# Basic search
pysearch find --pattern "requests.get" --path . --regex --context 3

# Advanced search with filters
pysearch find \
  --path src tests \
  --include "**/*.py" \
  --exclude "*/.venv/*" "*/build/*" \
  --pattern "def .*_handler" \
  --regex \
  --context 4 \
  --format json \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache"
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

### Development Commands

```bash
# Run all validation checks
make validate

# Run tests with coverage
make test

# Run linters
make lint

# Format code
make format

# Run type checking
make type

# Run benchmarks
make bench
```

## MCP (Model Context Protocol) Integration

PySearch includes several MCP server implementations for LLM integration:

- **Main MCP Server**: Advanced features with fuzzy search, analysis, and composition
- **Basic MCP Server**: Core search functionality (legacy)
- **FastMCP Server**: Optimized performance implementation

Run an MCP server:
```bash
./scripts/run-mcp-server.sh main
```

## Development Conventions

### Code Style

- Follow PEP 8 with a line length of 100 characters
- Use type hints throughout the codebase
- Use dataclasses for data structures
- Use enums for predefined constants
- Use Click for CLI interfaces
- Use docstrings for all public functions and classes

### Testing

- Use pytest for testing
- Maintain >90% test coverage
- Include both unit and integration tests
- Use property-based testing with hypothesis for complex logic
- Include benchmark tests for performance-critical code

### Documentation

- Use Google-style docstrings
- Maintain comprehensive README.md
- Keep CLI help texts up to date
- Document all public APIs
- Include usage examples

### Git Workflow

- Use conventional commits
- Maintain a clean commit history
- Write descriptive commit messages
- Keep pull requests focused on single features/fixes
- Ensure all CI checks pass before merging

## Key Files and Directories

### Configuration Files
- `pyproject.toml`: Project configuration, dependencies, build settings
- `Makefile`: Development commands and workflows
- `requirements.txt`: Core dependencies
- `requirements-dev.txt`: Development dependencies

### Scripts
- `scripts/dev-install.sh`: Development environment setup
- `scripts/run-mcp-server.sh`: Run MCP servers
- `scripts/validate-project.sh`: Project validation

### Documentation
- `docs/`: Comprehensive documentation in Markdown
- `README.md`: Project overview and quick start guide
- `CHANGELOG.md`: Release history and changes

## Common Development Tasks

### Adding a New Feature
1. Create a new branch
2. Implement the feature with tests
3. Update documentation
4. Run validation (`make validate`)
5. Commit with conventional commit message
6. Create pull request

### Running Specific Tests
```bash
# Run a specific test file
pytest tests/test_api_min.py

# Run tests matching a pattern
pytest -k "test_ast"

# Run with coverage
pytest --cov=pysearch
```

### Debugging
- Use the `--debug` flag in CLI commands
- Check logs in `.pysearch-cache/logs/`
- Use the `--stats` flag to see performance metrics