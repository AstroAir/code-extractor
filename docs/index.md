# pysearch Documentation

Welcome to pysearch - a high-performance, context-aware search engine for Python codebases that supports text/regex/AST/semantic search, providing both CLI and programmable API interfaces, designed for engineering-grade retrieval in large multi-module projects.

## What is pysearch?

pysearch is a powerful code search tool that goes beyond simple text matching. It understands code structure, provides semantic search capabilities, and offers intelligent ranking of results. Whether you're working on a small project or a large enterprise codebase, pysearch helps you find what you're looking for quickly and accurately.

## Key Features

### 🔍 **Multiple Search Modes**
- **Text Search**: Fast string matching with case sensitivity options
- **Regex Search**: Full regex pattern support with named groups
- **AST Search**: Structure-aware code search using Abstract Syntax Trees
- **Semantic Search**: Concept-based search using lightweight semantic analysis

### ⚡ **High Performance**
- Incremental indexing with intelligent caching
- Parallel search execution
- Memory-efficient processing of large codebases
- Smart directory pruning and file filtering

### 🎯 **Context-Aware Results**
- Configurable context lines around matches
- Intelligent result ranking and scoring
- Match deduplication and similarity clustering
- File metadata and author information

### 🛠️ **Developer-Friendly**
- Both CLI and Python API interfaces
- Multiple output formats (text, JSON, highlighted console)
- Comprehensive configuration options
- Integration with development workflows

### 📊 **Advanced Features**
- Multi-repository search capabilities
- Search history and analytics
- MCP (Model Context Protocol) integration
- Dependency analysis and refactoring suggestions

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```bash
# Find all function definitions
pysearch find --pattern "def " --path ./src

# Search with regex for handler functions
pysearch find --pattern "def.*handler" --regex --context 3

# AST-based search with filters
pysearch find --pattern "def" --ast --filter-func-name ".*handler"
```

### Python API

```python
from pysearch import PySearch, SearchConfig

# Create search engine
config = SearchConfig(paths=["."], include=["**/*.py"])
engine = PySearch(config)

# Perform search
results = engine.search("def main")
print(f"Found {len(results.items)} matches")
```

## Documentation Sections

### Getting Started
- [Installation Guide](installation.md) - Complete installation instructions
- [Quick Start](usage.md) - Get up and running in minutes
- [Configuration](configuration.md) - Comprehensive configuration guide

### User Guide
- [Search Types](../examples/tutorials/05_search_types.py) - Understanding different search modes
- [Performance Tuning](performance.md) - Optimize for your use case
- [CLI Reference](cli-reference.md) - Complete command-line reference

### API Reference
- [PySearch API](api/pysearch.md) - Main search engine interface
- [Configuration API](api/config.md) - Configuration management
- [Types and Data Structures](api/types.md) - Core data types

### Advanced Topics
- [Architecture](architecture.md) - System design and components
- [MCP Integration](mcp-overview.md) - Model Context Protocol support
- [Multi-Repository Search](api/multi_repo.md) - Search across multiple codebases

### Help & Support
- [FAQ](faq.md) - Frequently asked questions
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Maintenance Guide](maintenance.md) - Project cleanup and maintenance procedures
- [Examples](../examples/README.md) - Comprehensive examples

## Use Cases

pysearch is perfect for:

- **Code Navigation**: Quickly find functions, classes, and variables
- **Refactoring**: Identify all usages of code elements
- **Code Review**: Search for patterns and anti-patterns
- **Documentation**: Find examples and usage patterns
- **Learning**: Explore unfamiliar codebases
- **Debugging**: Locate error sources and related code

## Community and Support

- **GitHub**: [pysearch repository](https://github.com/your-org/pysearch)
- **Issues**: [Report bugs and request features](https://github.com/your-org/pysearch/issues)
- **Contributing**: [Contribution guidelines](../CONTRIBUTING.md)

## License

pysearch is released under the MIT License. See [LICENSE](../LICENSE) for details.
