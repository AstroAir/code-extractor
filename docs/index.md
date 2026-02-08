# pysearch Documentation

Welcome to pysearch - a high-performance, context-aware search engine for Python codebases that supports text/regex/AST/semantic/fuzzy/boolean/GraphRAG search, providing CLI, Python API, and MCP (Model Context Protocol) interfaces, designed for engineering-grade retrieval in large multi-module projects.

## What is pysearch?

pysearch is a powerful code search tool that goes far beyond simple text matching. It understands code structure through AST analysis, provides semantic search with optional transformer-based embeddings, supports fuzzy matching for typo-tolerant queries, builds knowledge graphs for intelligent code navigation via GraphRAG, and offers multi-repository search across large codebases. Whether you're working on a small project or a large enterprise monorepo, pysearch helps you find what you're looking for quickly and accurately.

## Key Features

### 🔍 **Multiple Search Modes**

- **Text/Regex Search**: Enhanced regex via `regex` library with multiline mode and named groups
- **AST Search**: Structure-aware code search using Python Abstract Syntax Trees with function/class/decorator/import filters
- **Semantic Search**: Lightweight symbolic analysis + optional transformer-based embedding search
- **Fuzzy Search**: Multiple algorithms (Levenshtein, Jaro-Winkler, n-gram) with typo tolerance via `rapidfuzz`
- **Boolean Search**: Logical operators (`AND`, `OR`, `NOT`) for complex query composition
- **GraphRAG Search**: Graph Retrieval-Augmented Generation combining knowledge graphs with vector similarity

### ⚡ **High Performance**

- Multi-level caching (in-memory + disk) with LRU eviction and TTL
- Parallel search execution with configurable worker pools
- Content-addressed indexing (SHA256) with incremental updates
- Multi-index architecture: code snippets, full-text (SQLite FTS5), chunk, and vector indexes
- Distributed indexing for large codebases across multiple processes
- Smart directory pruning and file filtering

### 🎯 **Context-Aware Results**

- Configurable context lines around matches
- Pluggable ranking strategies with result deduplication and similarity clustering
- File metadata and author information
- Search history with bookmarks, sessions, and analytics

### 🛠️ **Developer-Friendly**

- CLI (Click-based), Python API, and MCP server interfaces
- Multiple output formats (text, JSON, syntax-highlighted console)
- IDE integration: jump-to-definition, find references, completions, hover info, diagnostics
- Real-time file monitoring with automatic index updates
- Comprehensive configuration via API, CLI, TOML files, and environment variables

### 📊 **Analysis & Intelligence**

- **GraphRAG Engine**: Entity extraction, relationship mapping, knowledge graph construction
- **Dependency Analysis**: Import graph generation, circular dependency detection, coupling metrics
- **Language Support**: Automatic detection and tree-sitter processing for 20+ languages
- **Vector Database Support**: LanceDB, Qdrant, and Chroma backends with multiple embedding providers
- **Performance Monitoring**: Real-time profiling, metrics collection, and optimization suggestions
- **Advanced Error Handling**: Circuit breaker pattern, recovery manager, error aggregation

## Quick Start

### Installation

```bash
pip install -e .

# Optional features
pip install -e ".[graphrag]"    # GraphRAG and enhanced analysis
pip install -e ".[vector]"      # Vector database support
pip install -e ".[semantic]"    # Transformer-based semantic search
pip install -e ".[watch]"       # File watching support
pip install -e ".[all]"         # All optional features
```

### Basic Usage

```bash
# Find all function definitions
pysearch find --pattern "def " --path ./src

# Search with regex for handler functions
pysearch find --pattern "def.*handler" --regex --context 3

# AST-based search with filters
pysearch find --pattern "def" --ast --filter-func-name ".*handler"

# Semantic search
pysearch find --pattern "database connection" --semantic --context 5

# Boolean queries
pysearch find --pattern "(async AND handler) NOT test" --logic
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

- [Installation Guide](getting-started/installation.md) - System requirements and all installation methods
- [Quick Start](getting-started/quickstart.md) - Get up and running in minutes
- [Configuration](guide/configuration.md) - Comprehensive configuration guide

### User Guide

- [CLI Reference](guide/cli-reference.md) - Complete command-line reference
- [Performance Tuning](guide/performance.md) - Optimize for your use case
- [Style Guide](development/style-guide.md) - Code style conventions

### API Reference

- [PySearch API](api/pysearch.md) - Main search engine interface
- [Configuration API](api/config.md) - Configuration management
- [Types and Data Structures](api/types.md) - Core data types
- [Full API Reference](api/index.md) - Complete API documentation

### Advanced Topics

- [Architecture](advanced/architecture.md) - System design and component internals
- [Advanced Features](advanced/features.md) - Enhanced indexing engine
- [GraphRAG Guide](advanced/graphrag.md) - Knowledge graph search
- [Advanced Indexing](advanced/indexing-guide.md) - Indexing engine deep dive
- [Indexing Architecture](advanced/indexing-architecture.md) - Enhanced indexing design
- [MCP Integration](mcp/index.md) - Model Context Protocol support
- [MCP Tutorial](mcp/tutorial.md) - Step-by-step MCP setup
- [MCP API Reference](mcp/api.md) - MCP tool documentation
- [MCP Advanced](mcp/advanced.md) - Advanced MCP capabilities
- [MCP Server Guide](mcp/server-guide.md) - MCP server deployment

### Help & Support

- [FAQ](help/faq.md) - Frequently asked questions
- [Troubleshooting](help/troubleshooting.md) - Common issues and solutions
- [Maintenance Guide](development/maintenance.md) - Project cleanup and maintenance procedures
- [Roadmap](development/roadmap.md) - Future plans and priorities

## Use Cases

pysearch is perfect for:

- **Code Navigation**: Quickly find functions, classes, and variables across large codebases
- **Refactoring**: Identify all usages of code elements with dependency analysis
- **Code Review**: Search for patterns, anti-patterns, and security issues
- **Architecture Analysis**: Visualize dependencies and detect circular imports
- **Documentation**: Find examples and usage patterns with semantic search
- **Learning**: Explore unfamiliar codebases with GraphRAG knowledge graphs
- **Debugging**: Locate error sources and related code with fuzzy matching
- **LLM Integration**: Provide code context to AI assistants via MCP servers

## Community and Support

- **GitHub**: [pysearch repository](https://github.com/AstroAir/pysearch)
- **Issues**: [Report bugs and request features](https://github.com/AstroAir/pysearch/issues)
- **Contributing**: [Contribution guidelines](../CONTRIBUTING.md)

## License

pysearch is released under the MIT License. See [LICENSE](../LICENSE) for details.
