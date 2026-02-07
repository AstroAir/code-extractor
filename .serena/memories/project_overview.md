# PySearch Project Overview

## Purpose
**PySearch** is a high-performance, context-aware search engine for Python codebases that provides engineering-grade retrieval capabilities through multiple search modes (text/regex/AST/semantic), intelligent caching, and extensible architecture.

### Core Goals
- **Performance**: Optimized for large multi-module projects with parallel processing
- **Flexibility**: Multiple search strategies from simple text to advanced semantic analysis
- **Developer Experience**: Both CLI and Python API interfaces
- **Extensibility**: Plugin-ready architecture for custom matchers and integrations

## Tech Stack

### Language & Runtime
- **Python**: 3.10+ required
- **Package Manager**: pip (with uv.lock support)

### Core Dependencies
- `regex`: Enhanced regex support
- `rich`/`pygments`: Terminal highlighting
- `orjson`: Fast JSON serialization
- `click`: CLI framework
- `pydantic`: Config validation
- `fastmcp`: MCP server support
- `qdrant-client`: Vector database for semantic search

### Optional Dependencies
- `[semantic]`: Advanced semantic search with transformers
- `[graphrag]`: Knowledge graph capabilities (requires Qdrant)
- `[vector]`: Vector database support (Qdrant, FAISS)
- `[dev]`: Testing, linting, typing tools (pytest, mypy, ruff, black)

### Development Tools
- **Testing**: pytest with coverage (>85% target)
- **Linting**: ruff (fast Python linter)
- **Formatting**: black (100 character line length)
- **Type Checking**: mypy (strict for new code)
- **Documentation**: MkDocs with Material theme

## Key Features
- Multiple search modes: text/regex/AST/semantic
- Intelligent caching and indexing
- MCP (Model Context Protocol) server integration
- Multi-repository search support
- GraphRAG knowledge graph capabilities
- Comprehensive error handling and logging
