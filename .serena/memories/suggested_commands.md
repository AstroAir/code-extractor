# Suggested Commands

## System Utilities (Windows)
Since this project is on Windows, use these commands:
- File listing: `dir` or `ls`
- Directory change: `cd`
- File search: `findstr` or use git bash
- Process management: `tasklist`, `taskkill`
- Environment: `set`, `echo %VAR%`

## Setup and Installation
```bash
# Install package only
python -m pip install -e .

# Install with dev dependencies
python -m pip install -e ".[dev]"

# Alternative: use Make
make install       # Install package only
make dev          # Install dev dependencies
```

## Testing Commands
```bash
# Run all tests with coverage
pytest
# Or using make
make test

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Exclude slow tests
pytest -m benchmark      # Benchmark tests only

# Run specific test file
pytest tests/test_specific.py::test_function

# Run benchmarks
pytest -q -k benchmark
# Or using make
make bench
```

## Code Quality Commands
```bash
# Linting
ruff check .
black --check .
# Or using make
make lint

# Auto-fix formatting
black .
ruff check . --fix
# Or using make
make format

# Type checking
mypy src/
# Or using make
make type

# Full validation (lint + type + test + structure check)
make validate
```

## Running the Application
```bash
# CLI usage
pysearch find --pattern "def main" --path . --regex

# Python API example
python -c "from pysearch.api import PySearch; from pysearch.config import SearchConfig; engine = PySearch(SearchConfig(paths=['.'])); print(engine.search('def main'))"
```

## Documentation
```bash
# Build documentation
make docs
# Or
mkdocs build --clean --strict

# Serve documentation locally
make docs-serve
# Or
mkdocs serve -a 0.0.0.0:8000

# Clean documentation artifacts
make docs-clean
```

## MCP Servers
```bash
# Test MCP servers
make mcp-servers
```

## Cache and Cleanup
```bash
# Clean all cache files and build artifacts
make clean
# This removes:
# - build/, dist/, sdist/, wheels/
# - .pytest_cache/, .mypy_cache/, .ruff_cache/
# - .pysearch-cache/, .coverage/, htmlcov/
# - site/, .mkdocs_cache/, docs_build/
# - All __pycache__ directories
```

## Git Commands
```bash
# Standard git workflow
git status
git add .
git commit -m "message"
git push

# View recent commits
git log --oneline -10
```

## Project Structure Check
```bash
# Validate project structure
make check-structure
```
