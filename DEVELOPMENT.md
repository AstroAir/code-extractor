# PySearch Development Guide

This guide covers the development setup, project structure, and contribution workflow for PySearch.

## Quick Setup

```bash
# Clone and setup
git clone <repository-url>
cd pysearch
./scripts/dev-install.sh
```

## Project Structure

```text
├── src/pysearch/          # Core PySearch library
│   ├── __init__.py        # Package initialization
│   ├── api.py             # Main API interface
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   └── ...                # Other core modules
├── mcp/                   # MCP (Model Context Protocol) servers
│   ├── servers/           # MCP server implementations
│   │   ├── mcp_server.py  # Main MCP server with advanced features
│   │   ├── basic_mcp_server.py # Basic/legacy MCP server
│   │   └── ...            # Other server implementations
│   ├── shared/            # Shared MCP utilities
│   │   ├── composition.py # Search composition utilities
│   │   ├── progress.py    # Progress reporting
│   │   └── ...            # Other shared components
│   └── README.md          # MCP-specific documentation
├── tools/                 # Development and utility tools
├── examples/              # Usage examples and demos
├── tests/                 # Test suite
├── docs/                  # Documentation (MkDocs)
├── scripts/               # Build and development scripts
│   ├── dev-install.sh     # Development setup
│   ├── validate-project.sh # Project validation
│   └── run-mcp-server.sh  # MCP server runner
├── configs/               # Configuration files
└── .venv/                 # Virtual environment (development)
```

### Clean Development Environment

The project maintains a clean development environment:
- **Cache directories** (`.mypy_cache`, `.pytest_cache`, etc.) are automatically managed
- **Build artifacts** are excluded and regenerated as needed
- **Single virtual environment** (`.venv/`) for consistent development
- **No duplicate environments** or unnecessary files

## Development Workflow

### 1. Setup Development Environment

```bash
# Install development dependencies
./scripts/dev-install.sh

# Or manually:
pip install -e ".[dev]"
pre-commit install
```

### 2. Code Quality Checks

```bash
# Run all validation checks
make validate

# Individual checks
make lint      # Linting with ruff and black
make type      # Type checking with mypy
make test      # Run test suite
make format    # Auto-format code
```

### 3. Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=pysearch

# Run benchmarks
make bench
```

### 4. MCP Server Development

```bash
# Test MCP servers
make mcp-servers

# Run specific MCP server
./scripts/run-mcp-server.sh main

# Available servers: basic, main, fastmcp, pysearch
```

### 5. Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make serve

# Documentation will be available at http://localhost:8000
```

## Code Style

- **Python**: Follow PEP 8, enforced by `black` and `ruff`
- **Line Length**: 100 characters
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google-style docstrings for all public functions/classes

## Testing Guidelines

- **Coverage**: Maintain >85% test coverage
- **Test Structure**: Mirror the `src/` structure in `tests/`
- **Naming**: Test files should be named `test_<module>.py`
- **Fixtures**: Use pytest fixtures for common test data

## Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes following the code style
4. **Test** your changes: `make validate`
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to the branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

## Release Process

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Run validation
make validate

# Build and check
make release
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've installed in editable mode: `pip install -e .`
2. **MCP Server Issues**: Check that all dependencies are installed: `pip install -e ".[dev]"`
3. **Test Failures**: Run `make clean` to clear caches, then `make test`
4. **Cache Issues**: All cache directories are automatically managed and regenerated when needed
5. **Virtual Environment**: Use `.venv/` for consistent development environment

### Getting Help

- Check existing issues in the repository
- Run `make help` for available commands
- Review the documentation in `docs/`
