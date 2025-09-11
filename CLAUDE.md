# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

**Setup and Installation:**
```bash
make dev           # Install dev dependencies (includes pytest, mypy, ruff, black)
make install       # Install package only
```

**Testing:**
```bash
make test          # Run pytest with coverage (also: pytest)
make bench         # Run benchmarks (pytest -q -k benchmark)
```

**Code Quality:**
```bash
make lint          # Run ruff check + black --check
make format        # Run black + ruff --fix
make type          # Run mypy on src and tests
```

**Alternative script usage:**
```bash
./scripts/test.sh  # Simple test runner
./scripts/lint.sh  # Combined lint + format check + type check
```

## Architecture Overview

**Core Components:**
- `pysearch.indexer`: File scanning, hash/mtime caching, incremental indexing
- `pysearch.matchers`: Text/regex matching, AST node filtering, semantic signal extraction
- `pysearch.scorer`: Result scoring and ranking based on multiple signals
- `pysearch.formatter`: Output rendering (text/json/highlight formats)
- `pysearch.api`: Main PySearch API class, unified entry point
- `pysearch.cli`: Click-based command line interface
- `pysearch.config`: SearchConfig object definition and validation

**Main Entry Points:**
- CLI: `pysearch find` command via `pysearch.cli:main`
- API: `PySearch` class from `pysearch.api`

**Data Flow:**
1. CLI parses args → constructs `SearchConfig` and `Query`
2. `PySearch.run()` coordinates indexer scanning and matcher execution
3. Results scored/ranked by `scorer`
4. Output formatted by `formatter`

**Key Dependencies:**
- `regex`: Enhanced regex support
- `rich`/`pygments`: Terminal highlighting
- `orjson`: Fast JSON serialization
- `click`: CLI framework
- `pydantic`: Config validation

## Testing and Coverage

- Target coverage: >85% (configured in pyproject.toml)
- Test structure: `tests/` directory with comprehensive test suite
- Benchmarks: `tests/benchmarks/` for performance testing
- Run single test: `pytest tests/test_specific.py::test_function`

## Configuration

- Package config: `pyproject.toml` (includes pytest, coverage, ruff, black, mypy settings)
- Line length: 100 characters (black + ruff)
- Python version: 3.10+ required
- Type checking: mypy with strict settings for new code

## Build and Release

```bash
make clean         # Clean build artifacts and cache files
make release       # Build package (requires TWINE_* env vars for upload)
```

## Project Maintenance

**Cache Management:**
- Cache directories (`.mypy_cache`, `.pytest_cache`, `.pysearch-cache`, etc.) are automatically managed
- Use `make clean` to remove all cache files and build artifacts
- Cache files are regenerated automatically when needed

**Virtual Environment:**
- Project uses `.venv/` for development environment
- No duplicate virtual environments after cleanup