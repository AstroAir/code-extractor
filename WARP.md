# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: pysearch — a high-performance, context-aware search engine for codebases with text/regex/AST/semantic search via CLI and Python API.

Key commands (cross-platform)
- Environment setup
  - Python 3.10+
  - Install (editable) and dev tools:
    - python -m pip install -U pip
    - python -m pip install -e .
    - python -m pip install -e ".[dev]"
    - python -m pip install pre-commit; pre-commit install
  - Validate setup (composite):
    - make validate
    - If make isn’t available:
      - python -m ruff check .
      - python -m black --check .
      - python -m mypy
      - python -m pytest

- Build, lint, type-check, format
  - Lint: python -m ruff check .
  - Format: python -m black . && python -m ruff check . --fix
  - Type-check: python -m mypy
  - Clean artifacts (Unix): make clean

- Tests
  - Run all tests: python -m pytest
  - With coverage: python -m pytest --cov=pysearch
  - Benchmarks: python -m pytest -q -k benchmark
  - Run a single test:
    - python -m pytest tests/unit/api/test_api_min.py::test_basic_api
    - Or by expression: python -m pytest -k "api and not slow"
  - Useful pytest config is in pyproject.toml (markers like unit, integration, e2e, benchmark; asyncio_mode auto; addopts presets).

- CLI usage (installed as entry point: pysearch)
  - Basic:
    - pysearch find --pattern "def " --path ./src --include "**/*.py"
  - Regex with context and JSON:
    - pysearch find --pattern "def.*handler" --regex --context 3 --format json
  - AST filters:
    - pysearch find --pattern def --ast --filter-func-name ".*handler" --filter-decorator "lru_cache"
  - Semantic hinting:
    - pysearch find --pattern "database connection" --semantic --context 5

- Documentation
  - Build docs: mkdocs build --clean --strict
  - Serve docs: mkdocs serve -a 0.0.0.0:8000
  - Makefile equivalents: make docs, make docs-serve, make docs-check

- MCP servers
  - Quick run (main server): python mcp/servers/mcp_server.py
  - Other servers: python mcp/servers/basic_mcp_server.py; python mcp/servers/fastmcp_server.py
  - Helper script (Unix): ./scripts/run-mcp-server.sh main

- Makefile shortcuts (Unix/macOS)
  - make dev       # install dev deps
  - make lint      # ruff + black --check
  - make format    # black + ruff --fix
  - make type      # mypy
  - make test      # pytest
  - make bench     # benchmarks
  - make docs / make docs-serve / make docs-check
  - make mcp-servers  # import-check MCP servers
  - make validate  # lint + type + test + structure

High-level architecture (big picture)
- Interfaces
  - CLI: src/pysearch/cli/main.py exposes pysearch find with rich flags for pattern types, AST filters, metadata filters, output formats, and ranking options.
  - Python API: src/pysearch/core/api.py provides class PySearch orchestrating indexing, matching, scoring, formatting, and optional integrations (GraphRAG, metadata indexing, multi-repo, file watching, caching, parallel search).

- Core engine
  - Configuration: src/pysearch/core/config.py defines SearchConfig and ranking knobs, language/file filters, performance toggles (parallelism, cache, dir pruning), and optional vector/GraphRAG settings.
  - Types: src/pysearch/core/types.py re-exports structured types (Query, SearchResult, SearchItem, ASTFilters, OutputFormat, GraphRAGQuery, etc.) for stable public imports.
  - Indexing: src/pysearch/indexing/ provides Indexer (file discovery, include/exclude resolution, metadata caching, incremental updates) and cache_manager for on-disk index state.
  - Search: src/pysearch/search/ implements concrete matchers for text/regex (matchers.py), AST (via Python ast and filters), fuzzy (rapidfuzz/levenshtein), and semantic variants; scorer.py handles ranking, de-duplication, and optional clustering.
  - Utilities: src/pysearch/utils/ holds error handling, logging, file watching, metadata filters, formatting/highlighting, performance monitors.
  - Storage: src/pysearch/storage/ offers optional vector DB clients (qdrant_client.py) for semantic/GraphRAG use cases.

- Orchestration and data flow (end-to-end)
  1) CLI parses args → builds SearchConfig and Query → instantiates PySearch
  2) Indexer scans paths, applies includes/excludes, updates cache
  3) For each candidate file: matchers run (text/regex/AST/semantic/fuzzy) → produce SearchItem spans + context
  4) Scorer ranks and deduplicates → Results aggregated with stats
  5) Formatter renders output (text/JSON/highlight) → CLI prints and optional stats
  Optional: integrations enable
  - GraphRAG: knowledge graph building/query via Qdrant (when configured)
  - Enhanced indexing: semantic/complexity/dependency indexing for faster queries
  - File watching: reactive re-indexing
  - Multi-repo: coordinated search across logical repositories

Repository-specific notes
- Configuration authority: pyproject.toml centralizes pytest, coverage, black, ruff, and mypy settings (line length 100, Python 3.10+). Respect addopts and markers from [tool.pytest.ini_options].
- CLI entry point: [project.scripts] maps pysearch → pysearch.cli:main.
- Docs: MkDocs site via mkdocs.yml; docs directory contains architecture and usage references.
- Scripts (Unix): scripts/dev-install.sh, scripts/validate-project.sh, scripts/run-mcp-server.sh streamline setup, validation, and MCP server runs.
- Tests: tests/ contains unit, integration, performance suites; leverage markers (e.g., -m unit, -m "not slow").

Pulling in important rules from CLAUDE.md (applies here too)
- Preferred commands mirror Makefile targets and Python module invocations listed above.
- Single test execution pattern: pytest tests/<path>::test_name; you can also use -k expressions.
- Coverage, lint, type, and format commands match the ones documented in this file.

Quick recipes
- Run a focused API unit test:
  - python -m pytest tests/unit/api/test_api_min.py::test_api_min
- Lint, type-check, then test:
  - python -m ruff check . && python -m mypy && python -m pytest
- Search for async defs across src and tests:
  - pysearch find --path src tests --include "**/*.py" --regex --pattern "async def|await " --format highlight

Troubleshooting hints (project-specific)
- If imports fail, ensure editable install: python -m pip install -e .
- If cached state causes odd test results, clear artifacts (Unix): make clean, then re-run tests.
- On Windows without make, invoke underlying python -m commands shown above.

