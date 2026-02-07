# Codebase Structure

## Top-Level Organization
```
code-extractor/
├── src/pysearch/          # Core PySearch library
│   ├── cli/               # Command-line interface
│   ├── core/              # Main API, configuration, types
│   │   ├── types/         # Type definitions
│   │   ├── history/       # Search history management
│   │   └── integrations/  # Core integrations
│   ├── indexing/          # File scanning, caching, metadata
│   │   ├── advanced/      # Advanced indexing features
│   │   ├── cache/         # Cache management
│   │   ├── indexes/       # Index implementations
│   │   └── metadata/      # Metadata indexing
│   ├── search/            # Pattern matching algorithms
│   ├── analysis/          # Code analysis, language detection
│   │   └── graphrag/      # GraphRAG knowledge graph
│   ├── utils/             # Utilities, error handling, logging
│   ├── storage/           # Vector database integration
│   └── integrations/      # External system integrations
│       ├── multi_repo.py
│       └── distributed_indexing.py
├── mcp/                   # MCP (Model Context Protocol) servers
│   ├── servers/           # MCP server implementations
│   │   ├── basic_mcp_server.py
│   │   └── enhanced_fastmcp_server_simple.py
│   └── shared/            # Shared MCP utilities
├── tests/                 # Test suite
│   ├── unit/              # Fast, isolated component tests
│   │   ├── api/
│   │   ├── cli/
│   │   ├── core/
│   │   └── search/
│   └── integration/       # Component interaction tests
├── docs/                  # Documentation
├── scripts/               # Build and development scripts
├── configs/               # Configuration files
├── .venv/                 # Virtual environment (development)
├── pyproject.toml         # Package configuration
├── Makefile               # Development commands
├── CLAUDE.md              # AI context documentation
└── README.md              # Project overview
```

## Key Modules and Responsibilities

### Core (`src/pysearch/core/`)
- **api.py**: Main PySearch API entry point
- **config.py**: SearchConfig class for configuration
- **types/**: Type definitions (Query, Result, Filter, etc.)
- **history/**: Search history, bookmarks, sessions, analytics

### Indexing (`src/pysearch/indexing/`)
- **indexer.py**: Main file indexing logic
- **cache_manager.py**: Cache management
- **metadata/**: Metadata indexing and database
- **indexes/**: Various index implementations (chunk, code_snippets, full_text, vector)
- **advanced/**: Advanced indexing features (chunking, coordinator, engine, integration)

### Search (`src/pysearch/search/`)
- **matchers.py**: Pattern matchers (text, regex, AST, semantic)
- **scorer.py**: Result scoring and ranking
- **fuzzy.py**: Fuzzy search capabilities
- **semantic_advanced.py**: Advanced semantic search

### Analysis (`src/pysearch/analysis/`)
- **dependency_analysis.py**: Dependency analysis
- **language_detection.py**: Language detection
- **language_support.py**: Multi-language support
- **graphrag/**: GraphRAG knowledge graph implementation

### CLI (`src/pysearch/cli/`)
- **main.py**: CLI entry point
- **__main__.py**: Python module execution

### Utils (`src/pysearch/utils/`)
- **error_handling.py**: Error handling utilities
- **formatter.py**: Output formatting
- **file_watcher.py**: File watching for auto-indexing
- **logging_config.py**: Logging configuration
- **performance_monitoring.py**: Performance monitoring

### Storage (`src/pysearch/storage/`)
- **qdrant_client.py**: Qdrant vector database client
- **vector_db.py**: Vector database abstraction

### MCP (`mcp/`)
- **servers/**: MCP server implementations for LLM integration
- **shared/**: Shared utilities (progress, validation, resource/session management)

## Data Flow
1. User (CLI/API) creates SearchConfig
2. PySearch engine initializes
3. Indexer scans and caches files
4. User executes search query
5. Matchers run in parallel (text/regex/AST/semantic)
6. Scorer ranks results
7. Formatter outputs results

## Module Dependencies
- CLI depends on Core
- Core depends on Indexing, Search, Utils
- Indexing depends on Utils
- Search depends on Analysis
- Analysis depends on Core
- Storage is used by Indexing and Search
- MCP depends on Core and Indexing
