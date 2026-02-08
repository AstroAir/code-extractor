# Code Style and Conventions

## Style Guidelines
- **Line length**: 100 characters (black + ruff)
- **Import order**: stdlib → third-party → local (with isort)
- **Type hints**: Required for public APIs
- **Docstrings**: Google-style for all public functions/classes
- **Naming conventions**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

## Code Quality Standards

### Linting and Formatting
```bash
# Check linting
ruff check .
black --check .

# Auto-fix
ruff check . --fix
black .
```

### Type Checking
```bash
# Run mypy on src and tests
mypy src/
mypy tests/
```

## Key Patterns to Follow

### Configuration
- All configuration goes through `SearchConfig` class
- Use pydantic for config validation

### Error Handling
- Use `ErrorCollector` for comprehensive error tracking
- Proper exception handling with specific error types

### Logging
- Use the `SearchLogger` from `utils.logging_config`
- Structured logging with appropriate levels

### Type Safety
- Add type hints to all public APIs
- Use mypy strict mode for new code
- Prefer type annotations over comments

## Architecture Principles
- **Modularity**: Clear separation between core, indexing, search, analysis, CLI, utils, storage, integrations, and MCP modules
- **Configuration-driven**: All behavior configurable through SearchConfig
- **Performance-first**: Optimized for large codebases with caching
- **Testability**: Comprehensive test coverage with clear test organization
