# Contributing Guide

Thank you for your interest in contributing to pysearch! We welcome contributions from the community and appreciate your help in making this project better.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Branching and Commit Guidelines](#branching-and-commit-guidelines)
- [Code Style and Quality](#code-style-and-quality)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **Git** for version control
- **Basic knowledge** of Python development
- **Familiarity** with command line tools

### Quick Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/your-username/pysearch.git
   cd pysearch
   ```

3. **Set up development environment** (see below)
4. **Create a feature branch** for your changes
5. **Make your changes** and test them
6. **Submit a pull request**

---

## Development Environment

### Initial Setup

1. **Install Python 3.10+**

   ```bash
   # Check your Python version
   python --version  # Should be 3.10+
   ```

2. **Install development dependencies**

   ```bash
   # Upgrade pip first
   python -m pip install -U pip

   # Install pysearch in development mode with dev dependencies
   python -m pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**

   ```bash
   # Install pre-commit
   python -m pip install pre-commit

   # Set up git hooks
   pre-commit install
   ```

4. **Verify installation**

   ```bash
   # Run validation script
   make validate

   # Or manually check components
   python -c "import pysearch; print('✅ pysearch imported successfully')"
   pytest --version
   ruff --version
   mypy --version
   ```

### Development Commands

We use a Makefile to simplify common development tasks:

```bash
# Development setup
make dev          # Install development dependencies
make clean        # Clean build artifacts and cache

# Code quality
make lint         # Run linting (ruff + black)
make format       # Auto-format code
make type         # Type checking (mypy)
make check        # Run all checks (lint + type + test)

# Testing
make test         # Run tests with coverage
make test-fast    # Run tests without coverage
make benchmark    # Run performance benchmarks

# Documentation
make docs         # Build documentation
make docs-serve   # Serve docs locally for development

# Validation
make validate     # Run full validation suite
```

### IDE Setup

#### VS Code

Recommended extensions:

- Python
- Pylance
- Ruff
- Black Formatter
- GitLens

Settings (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. **Set Python interpreter** to your virtual environment
2. **Enable** Ruff and Black in settings
3. **Configure** pytest as test runner
4. **Set up** pre-commit integration

---

## Branching and Commit Guidelines

### Branch Model

We use a **feature branch workflow**:

- **`main`**: Stable branch, only updated via pull requests
- **Feature branches**: `feat/<short-descriptive-name>`
- **Bug fix branches**: `fix/<short-descriptive-name>`
- **Documentation**: `docs/<short-descriptive-name>`
- **Build/CI**: `build/<short-descriptive-name>`, `ci/<short-descriptive-name>`

### Branch Naming Examples

```bash
# Good branch names
feat/ast-semantic-search
fix/memory-leak-large-files
docs/api-reference-update
build/github-actions-optimization

# Avoid
feature-branch
my-changes
fix
```

### Commit Message Format

We follow **Conventional Commits** specification:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Commit Types

- **`feat`**: New feature
- **`fix`**: Bug fix
- **`docs`**: Documentation changes
- **`style`**: Code style changes (formatting, no logic changes)
- **`refactor`**: Code refactoring (no functional changes)
- **`test`**: Adding or updating tests
- **`build`**: Build system or dependency changes
- **`ci`**: CI/CD configuration changes
- **`chore`**: Maintenance tasks

#### Examples

```bash
# Good commit messages
feat(api): add semantic search capability
fix(indexer): resolve memory leak in large file processing
docs(readme): update installation instructions
test(search): add comprehensive AST filter tests
refactor(config): simplify SearchConfig initialization

# Include scope when relevant
feat(cli): add --semantic flag for semantic search
fix(cache): handle cache corruption gracefully
```

### Creating a Feature Branch

```bash
# Start from main
git checkout main
git pull origin main

# Create and switch to feature branch
git checkout -b feat/your-feature-name

# Make your changes...
git add .
git commit -m "feat: add your feature description"

# Push to your fork
git push origin feat/your-feature-name
```

---

## Code Style and Quality

### Code Formatting

We use **Black** for code formatting with these settings:

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'
```

**Format your code:**

```bash
make format
# or
black .
```

### Linting

We use **Ruff** for fast Python linting:

```bash
make lint
# or
ruff check .
```

**Common linting rules:**

- Line length: 88 characters (Black compatible)
- Import sorting: isort compatible
- Docstring style: Google style
- Type hints: Required for public APIs

### Type Checking

We use **mypy** for static type checking:

```bash
make type
# or
mypy src/
```

**Type checking requirements:**

- All public APIs must have type hints
- Use `from __future__ import annotations` for forward references
- Prefer `list[str]` over `List[str]` (Python 3.10+ syntax)
- Use `typing.Protocol` for structural typing

### Code Quality Standards

#### Docstrings

Use **Google style** docstrings:

```python
def search_files(pattern: str, paths: list[str]) -> SearchResult:
    """Search for pattern in specified files.

    Args:
        pattern: The search pattern to match.
        paths: List of file paths to search.

    Returns:
        SearchResult containing matches and metadata.

    Raises:
        SearchError: If search operation fails.
        FileNotFoundError: If specified paths don't exist.

    Example:
        >>> result = search_files("def main", ["./src"])
        >>> print(f"Found {len(result.items)} matches")
    """
```

#### Error Handling

- Use specific exception types
- Provide helpful error messages
- Include context in error messages
- Handle edge cases gracefully

```python
# Good
try:
    content = file_path.read_text(encoding='utf-8')
except UnicodeDecodeError as e:
    raise EncodingError(
        f"Failed to decode {file_path}: {e}. "
        f"File may not be UTF-8 encoded."
    ) from e

# Avoid
try:
    content = file_path.read_text()
except Exception:
    pass  # Silent failure
```

#### Performance Considerations

- Use appropriate data structures
- Avoid premature optimization
- Profile before optimizing
- Document performance characteristics

---

## Testing

### Test Structure

We use **pytest** with this structure:

```text
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── benchmarks/     # Performance tests
├── fixtures/       # Test data
└── conftest.py     # Shared fixtures
```

### Writing Tests

#### Unit Tests

```python
import pytest
from pysearch import PySearch, SearchConfig
from pysearch.types import SearchResult

class TestPySearch:
    def test_basic_search(self):
        """Test basic text search functionality."""
        config = SearchConfig(paths=["./test_data"])
        engine = PySearch(config)

        result = engine.search("def main")

        assert isinstance(result, SearchResult)
        assert len(result.items) > 0
        assert result.stats.files_scanned > 0

    def test_search_with_invalid_path(self):
        """Test search with non-existent path."""
        config = SearchConfig(paths=["/nonexistent"])
        engine = PySearch(config)

        with pytest.raises(FileNotFoundError):
            engine.search("pattern")
```

#### Integration Tests

```python
def test_end_to_end_search_workflow(tmp_path):
    """Test complete search workflow."""
    # Create test files
    (tmp_path / "test.py").write_text("def main(): pass")

    # Configure and search
    config = SearchConfig(paths=[str(tmp_path)])
    engine = PySearch(config)
    result = engine.search("def main")

    # Verify results
    assert len(result.items) == 1
    assert "def main" in result.items[0].lines[0]
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_api.py

# Run with coverage
pytest --cov=pysearch

# Run benchmarks
pytest tests/benchmarks/ -k benchmark

# Run tests in parallel
pytest -n auto
```

### Test Guidelines

1. **Test naming**: Use descriptive names that explain what is being tested
2. **Test isolation**: Each test should be independent
3. **Test data**: Use fixtures for reusable test data
4. **Assertions**: Use specific assertions with clear messages
5. **Coverage**: Aim for >90% code coverage
6. **Performance**: Include performance regression tests

---

## Documentation

### Documentation Types

1. **API Documentation**: Docstrings in code
2. **User Guides**: Markdown files in `docs/`
3. **Examples**: Working code in `examples/`
4. **README**: Project overview and quick start

### Writing Documentation

#### Markdown Guidelines

- Use clear, concise language
- Include code examples
- Add table of contents for long documents
- Use consistent formatting
- Test all code examples

#### Code Examples

All code examples should be:

- **Runnable**: Test that examples work
- **Complete**: Include necessary imports
- **Realistic**: Use practical scenarios
- **Commented**: Explain complex parts

```python
# Good example
from pysearch import PySearch, SearchConfig

# Configure search for Python files only
config = SearchConfig(
    paths=["./src"],
    include=["**/*.py"],
    exclude=["**/__pycache__/**"]
)

# Create search engine
engine = PySearch(config)

# Search for function definitions
results = engine.search("def ", regex=False)
print(f"Found {len(results.items)} function definitions")
```

### Building Documentation

```bash
# Build documentation
make docs

# Serve locally for development
make docs-serve

# Check for broken links
make docs-check
```

---

## Pull Request Process

### Before Submitting

1. **Run all checks**:

   ```bash
   make validate  # Runs lint, type, test, and other checks
   ```

2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** if applicable
5. **Ensure commits follow** conventional commit format

### Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review** if applicable
5. **Final approval** and merge

### After Merge

- **Delete feature branch** (both local and remote)
- **Update local main** branch
- **Check release notes** if your change is included

---

## Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please read and follow these guidelines.

### Communication

- **Be respectful** and constructive
- **Ask questions** if something is unclear
- **Provide context** when reporting issues
- **Help others** when you can

### Getting Help

1. **Check documentation** first
2. **Search existing issues** for similar problems
3. **Ask in discussions** for general questions
4. **Create an issue** for bugs or feature requests

### Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

---

## Development Workflow Example

Here's a complete example of contributing a new feature:

```bash
# 1. Fork and clone
git clone https://github.com/your-username/pysearch.git
cd pysearch

# 2. Set up development environment
make dev
pre-commit install

# 3. Create feature branch
git checkout -b feat/semantic-search-improvements

# 4. Make changes
# ... edit files ...

# 5. Test your changes
make validate

# 6. Commit changes
git add .
git commit -m "feat(semantic): improve similarity scoring algorithm"

# 7. Push to your fork
git push origin feat/semantic-search-improvements

# 8. Create pull request on GitHub
# ... use the web interface ...

# 9. Address review feedback
# ... make additional commits ...

# 10. After merge, clean up
git checkout main
git pull origin main
git branch -d feat/semantic-search-improvements
```

---

## Questions?

If you have questions about contributing:

1. **Check the [FAQ](docs/faq.md)**
2. **Search [GitHub Discussions](https://github.com/your-org/pysearch/discussions)**
3. **Create a new discussion** for general questions
4. **Open an issue** for specific problems

Thank you for contributing to pysearch! 🚀

- 格式化：Black（100 列）
- Lint：Ruff（规则集：E,F,I,UP,B）
