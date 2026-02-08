# Frequently Asked Questions (FAQ)

This document answers common questions about pysearch usage, configuration, and troubleshooting.

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Usage & Features](#usage--features)
- [Performance & Optimization](#performance--optimization)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development & Contributing](#development--contributing)

---

## General Questions

### What is pysearch?

pysearch is a high-performance, context-aware search engine for Python codebases that supports text/regex/AST/semantic search. It provides both CLI and programmable API interfaces, designed for engineering-grade retrieval in large multi-module projects.

### How is pysearch different from grep or ripgrep?

While grep and ripgrep are excellent for text searching, pysearch offers:

- **AST-aware search**: Understand code structure (functions, classes, decorators)
- **Semantic search**: Find conceptually similar code
- **Context-aware results**: Intelligent context extraction
- **Multi-format output**: JSON, highlighted terminal, plain text
- **Python integration**: Native Python API for toolchain integration
- **Advanced filtering**: File metadata, language-specific filters

### What programming languages does pysearch support?

pysearch primarily focuses on Python but supports multiple languages:

- **Full support**: Python (.py, .pyx, .pyi)
- **Text search**: JavaScript, TypeScript, Java, C/C++, Rust, Go, and more
- **Planned**: Enhanced AST support for additional languages

### Is pysearch suitable for large codebases?

Yes! pysearch is designed for large codebases with:

- **Parallel processing**: Multi-threaded search execution
- **Intelligent caching**: File metadata and result caching
- **Incremental indexing**: Only processes changed files
- **Memory optimization**: Efficient memory usage patterns
- **Directory pruning**: Skip excluded directories early

---

## Installation & Setup

### What are the system requirements?

**Minimum**:

- Python 3.10+
- 512 MB RAM
- 50 MB disk space

**Recommended**:

- Python 3.11+
- 4 GB RAM
- Multi-core CPU
- SSD storage

### How do I install pysearch?

```bash
# From PyPI (when published)
pip install pysearch

# From source
pip install git+https://github.com/AstroAir/pysearch.git

# Development installation
git clone https://github.com/AstroAir/pysearch.git
cd pysearch
pip install -e ".[dev]"
```

### Can I use pysearch without installing it globally?

Yes! Use virtual environments:

```bash
# Create virtual environment
python -m venv pysearch-env
source pysearch-env/bin/activate  # Linux/macOS
# pysearch-env\Scripts\activate   # Windows

# Install pysearch
pip install pysearch
```

### How do I update pysearch?

```bash
# Update from PyPI
pip install --upgrade pysearch

# Update from source
pip install --upgrade git+https://github.com/AstroAir/pysearch.git
```

---

## Usage & Features

### How do I perform a basic search?

```bash
# CLI
pysearch find --pattern "def main" --path ./src

# Python API
from pysearch import PySearch, SearchConfig
engine = PySearch(SearchConfig(paths=["./src"]))
results = engine.search("def main")
```

### What's the difference between text, regex, and AST search?

- **Text search**: Simple string matching

  ```bash
  pysearch find --pattern "TODO"
  ```

- **Regex search**: Pattern matching with regular expressions

  ```bash
  pysearch find --pattern "def.*handler" --regex
  ```

- **AST search**: Structure-aware code search

  ```bash
  pysearch find --pattern "def" --ast --filter-func-name ".*handler"
  ```

### How do I search for specific code structures?

Use AST filters:

```bash
# Find functions with specific names
pysearch find --pattern "def" --ast --filter-func-name "test_.*"

# Find classes with decorators
pysearch find --pattern "class" --ast --filter-decorator "dataclass"

# Find specific imports
pysearch find --pattern "import" --ast --filter-import "requests.*"
```

### Can I search across multiple directories?

Yes:

```bash
# CLI
pysearch find --path ./src --path ./tests --pattern "pattern"

# API
config = SearchConfig(paths=["./src", "./tests"])
```

### How do I exclude certain files or directories?

```bash
# CLI
pysearch find --exclude "**/.venv/**" --exclude "**/build/**" --pattern "pattern"

# API
config = SearchConfig(exclude=["**/.venv/**", "**/build/**"])
```

### What output formats are available?

- **text**: Human-readable (default)
- **json**: Machine-readable structured data
- **highlight**: Syntax-highlighted terminal output

```bash
pysearch find --pattern "pattern" --format json
```

### How do I get more context around matches?

```bash
# Show 5 lines before and after each match
pysearch find --pattern "pattern" --context 5
```

---

## Performance & Optimization

### How can I make searches faster?

1. **Use specific paths**: Limit search scope

   ```bash
   pysearch find --path ./src --pattern "pattern"  # Not --path .
   ```

2. **Add exclusions**: Skip unnecessary directories

   ```bash
   pysearch find --exclude "**/.venv/**" --exclude "**/.git/**" --pattern "pattern"
   ```

3. **Enable parallel processing**:

   ```bash
   pysearch find --parallel --workers 8 --pattern "pattern"
   ```

4. **Disable unnecessary content**:

   ```bash
   pysearch find --no-docstrings --no-comments --pattern "pattern"
   ```

### Why is my first search slow?

The first search builds the file index. Subsequent searches are much faster due to caching. You can pre-build the index:

```python
from pysearch import PySearch, SearchConfig
engine = PySearch(SearchConfig(paths=["./src"]))
engine.indexer.build_index()  # Pre-build index
```

### How much memory does pysearch use?

Memory usage depends on:

- **Codebase size**: Larger codebases use more memory
- **File size limits**: Set limits to control memory usage
- **Parallel workers**: More workers use more memory
- **Context lines**: More context increases memory usage

Typical usage: 50-200 MB for medium codebases (10K-100K files).

### Can I limit memory usage?

Yes:

```python
config = SearchConfig(
    file_size_limit=1_000_000,  # 1MB file limit
    workers=2,                  # Fewer workers
    context=1                   # Less context
)
```

---

## Configuration

### How do I create a configuration file?

Create `pysearch.toml`:

```toml
[search]
paths = ["./src", "./tests"]
include = ["**/*.py"]
exclude = ["**/.venv/**", "**/__pycache__/**"]
context = 3
parallel = true

[content]
enable_docstrings = true
enable_comments = false
enable_strings = true
```

### Where should I put configuration files?

pysearch looks for configuration in:

1. `./pysearch.toml` (current directory)
2. `~/.config/pysearch/config.toml` (user config)
3. `/etc/pysearch/config.toml` (system config)

### Can I use environment variables?

Yes:

```bash
export PYSEARCH_PATHS="./src:./tests"
export PYSEARCH_PARALLEL="true"
export PYSEARCH_WORKERS="8"
export PYSEARCH_CONTEXT="5"
```

### How do I configure for different environments?

```python
# Development
dev_config = SearchConfig(
    paths=["./src"],
    parallel=True,
    strict_hash_check=False,  # Faster
    enable_comments=False     # Focus on code
)

# Production/CI
prod_config = SearchConfig(
    paths=["./src", "./tests"],
    parallel=True,
    strict_hash_check=True,   # More accurate
    enable_docstrings=True    # Include docs
)
```

---

## Troubleshooting

### No results found, but I know matches exist

1. **Check include/exclude patterns**:

   ```bash
   pysearch find --pattern "pattern" --include "**/*.py" --stats
   ```

2. **Verify paths**:

   ```bash
   pysearch find --pattern "pattern" --path ./correct/path
   ```

3. **Test with broader patterns**:

   ```bash
   pysearch find --pattern "def" --regex  # Should find functions
   ```

4. **Check content filters**:

   ```bash
   pysearch find --pattern "pattern" --docstrings --comments --strings
   ```

### Search is very slow

1. **Add more exclusions**:

   ```bash
   pysearch find --exclude "**/.venv/**" --exclude "**/.git/**" --pattern "pattern"
   ```

2. **Limit search scope**:

   ```bash
   pysearch find --path ./specific/directory --pattern "pattern"
   ```

3. **Enable parallel processing**:

   ```bash
   pysearch find --parallel --workers 4 --pattern "pattern"
   ```

4. **Disable unnecessary parsing**:

   ```bash
   pysearch find --no-docstrings --no-comments --pattern "pattern"
   ```

### Getting encoding errors

1. **Check file encoding**:

   ```python
   import chardet
   with open('file.py', 'rb') as f:
       result = chardet.detect(f.read())
       print(result['encoding'])
   ```

2. **Skip problematic files**:

   ```bash
   pysearch find --exclude "**/problematic_file.py" --pattern "pattern"
   ```

### Memory usage is too high

1. **Set file size limits**:

   ```python
   config = SearchConfig(file_size_limit=1_000_000)  # 1MB
   ```

2. **Reduce parallel workers**:

   ```python
   config = SearchConfig(workers=2)
   ```

3. **Decrease context lines**:

   ```python
   config = SearchConfig(context=1)
   ```

### Command not found after installation

1. **Check installation**:

   ```bash
   pip show pysearch
   python -m pysearch --version
   ```

2. **Add to PATH** (if needed):

   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

3. **Use full Python path**:

   ```bash
   python -m pysearch find --pattern "pattern"
   ```

---

## Development & Contributing

### How do I set up a development environment?

```bash
git clone https://github.com/AstroAir/pysearch.git
cd pysearch
pip install -e ".[dev]"
pre-commit install
pytest  # Run tests
```

### How do I run tests?

```bash
# All tests
pytest

# Specific test file
pytest tests/test_api.py

# With coverage
pytest --cov=pysearch

# Benchmarks
pytest tests/benchmarks -k benchmark
```

### How do I contribute?

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes and add tests**
4. **Run tests and linting**: `make test lint`
5. **Submit a pull request**

See [Contributing Guide](../../CONTRIBUTING.md) for details.

### How do I report bugs?

1. **Check existing issues** on GitHub
2. **Create a new issue** with:
   - pysearch version (`pysearch --version`)
   - Python version (`python --version`)
   - Operating system
   - Minimal reproduction example
   - Expected vs. actual behavior

### How do I request features?

1. **Check the roadmap** in [docs/roadmap.md](../development/roadmap.md)
2. **Search existing feature requests**
3. **Create a new issue** with:
   - Clear description of the feature
   - Use cases and benefits
   - Proposed implementation (if any)

---

## Still Have Questions?

If your question isn't answered here:

1. **Search the documentation**: Check other docs in the `docs/` directory
2. **Search GitHub issues**: Look for similar questions or issues
3. **Ask the community**: Start a discussion on GitHub
4. **Contact maintainers**: Create an issue for specific problems

## Contributing to FAQ

Found an error or have a question that should be added? Please:

1. **Edit this file** and submit a pull request
2. **Create an issue** suggesting the addition
3. **Start a discussion** about the topic

We appreciate community contributions to improve this FAQ!
