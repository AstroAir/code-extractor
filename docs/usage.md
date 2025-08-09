# Usage Guide

This guide covers how to use pysearch effectively, from basic searches to advanced features.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Search Types](#search-types)
- [Output Formats](#output-formats)
- [Advanced Features](#advanced-features)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python 3.10 or higher
- Operating System: Linux, macOS, or Windows

### Basic Installation

```bash
# Install from source
pip install -e .

# Or with pip (when published)
pip install pysearch
```

### Development Installation

For contributors and advanced users:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
make validate
```

### Verification

Test your installation:

```bash
# Check version
pysearch --version

# Run basic search
pysearch find --pattern "def main" --path .

# Run validation suite
./scripts/validate-project.sh
```

---

## Quick Start

### 30-Second Example

```bash
# Find all function definitions in Python files
pysearch find --pattern "def " --path ./src --include "**/*.py"

# Search with regex for handler functions
pysearch find --pattern "def.*handler" --regex --context 3

# Find class definitions with AST filtering
pysearch find --pattern "class" --ast --filter-class-name ".*Test"
```

### First Python Script

```python
from pysearch import PySearch, SearchConfig

# Create search engine
config = SearchConfig(paths=["./src"], include=["**/*.py"])
engine = PySearch(config)

# Perform search
results = engine.search("def main")

# Display results
for item in results.items:
    print(f"{item.file}: lines {item.start_line}-{item.end_line}")
```

---

## Command Line Interface

### Basic Syntax

```bash
pysearch find [OPTIONS] --pattern PATTERN
```

### Essential Options

#### Search Scope

```bash
# Single path
pysearch find --path ./src --pattern "pattern"

# Multiple paths
pysearch find --path ./src --path ./tests --pattern "pattern"

# Include specific file types
pysearch find --include "**/*.py" --include "**/*.pyx" --pattern "pattern"

# Exclude directories
pysearch find --exclude "**/.venv/**" --exclude "**/build/**" --pattern "pattern"
```

#### Search Modes

```bash
# Text search (default)
pysearch find --pattern "def main"

# Regex search
pysearch find --pattern "def.*handler" --regex

# AST structural search
pysearch find --pattern "def" --ast --filter-func-name "main"

# Semantic search (experimental)
pysearch find --pattern "database connection" --semantic
```

#### Output Control

```bash
# Control context lines
pysearch find --pattern "pattern" --context 5

# Choose output format
pysearch find --pattern "pattern" --format json
pysearch find --pattern "pattern" --format highlight

# Show performance statistics
pysearch find --pattern "pattern" --stats
```

### Advanced CLI Options

#### AST Filters

```bash
# Filter by function names
pysearch find --pattern "def" --ast --filter-func-name ".*handler"

# Filter by class names
pysearch find --pattern "class" --ast --filter-class-name "Test.*"

# Filter by decorators
pysearch find --pattern "def" --ast --filter-decorator "lru_cache"

# Filter by imports
pysearch find --pattern "import" --ast --filter-import "requests.*"

# Combine multiple filters
pysearch find --pattern "def" --ast \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache"
```

#### Content Filtering

```bash
# Skip docstrings
pysearch find --pattern "pattern" --no-docstrings

# Skip comments
pysearch find --pattern "pattern" --no-comments

# Skip string literals
pysearch find --pattern "pattern" --no-strings

# Search only in code
pysearch find --pattern "pattern" --no-docstrings --no-comments --no-strings
```

#### Performance Options

```bash
# Enable parallel processing
pysearch find --pattern "pattern" --parallel --workers 8

# Set file size limits
pysearch find --pattern "pattern" --max-file-size 1048576  # 1MB

# Enable caching
pysearch find --pattern "pattern" --cache --cache-ttl 3600
```

### Complete CLI Example

```bash
pysearch find \
  --path ./src --path ./tests \
  --include "**/*.py" \
  --exclude "**/.venv/**" "**/__pycache__/**" \
  --pattern "async def.*handler" \
  --regex \
  --context 4 \
  --format json \
  --filter-func-name ".*handler$" \
  --filter-decorator "lru_cache" \
  --no-docstrings \
  --parallel \
  --stats
```

---

## Python API

### Basic Usage

```python
from pysearch import PySearch, SearchConfig
from pysearch.types import Query, OutputFormat

# Create configuration
config = SearchConfig(
    paths=["./src"],
    include=["**/*.py"],
    exclude=["**/.venv/**"],
    context=3
)

# Initialize search engine
engine = PySearch(config)

# Simple search
results = engine.search("def main")

# Process results
for item in results.items:
    print(f"Found in {item.file}:")
    for line in item.lines:
        print(f"  {line}")
```

### Advanced API Usage

```python
from pysearch import PySearch, SearchConfig
from pysearch.types import Query, ASTFilters, MetadataFilters, Language

# Advanced configuration
config = SearchConfig(
    paths=["./src", "./tests"],
    include=["**/*.py", "**/*.pyx"],
    exclude=["**/.venv/**", "**/build/**"],
    context=5,
    parallel=True,
    workers=4,
    enable_docstrings=True,
    enable_comments=False,
    enable_strings=True
)

# Create search engine with caching
engine = PySearch(config)
engine.enable_caching(ttl=3600)

# Complex query with filters
ast_filters = ASTFilters(
    func_name=".*handler$",
    decorator="(lru_cache|cache)",
    class_name=".*Manager"
)

metadata_filters = MetadataFilters(
    min_lines=50,
    max_size=1024*1024,  # 1MB
    languages={Language.PYTHON},
    modified_after="2024-01-01"
)

query = Query(
    pattern="async def.*handler",
    use_regex=True,
    use_ast=True,
    context=5,
    output=OutputFormat.JSON,
    ast_filters=ast_filters,
    metadata_filters=metadata_filters,
    search_docstrings=False
)

# Execute search
results = engine.run(query)

# Analyze results
print(f"Found {len(results.items)} matches in {results.stats.elapsed_ms:.1f}ms")
print(f"Scanned {results.stats.files_scanned} files")

for item in results.items:
    print(f"\n{item.file} (score: {item.score:.2f}):")
    print(f"  Lines {item.start_line}-{item.end_line}")

    # Show match spans for highlighting
    for span in item.match_spans:
        line_idx, (start_col, end_col) = span
        actual_line = item.start_line + line_idx
        print(f"    Match at line {actual_line}, columns {start_col}-{end_col}")
```

### API Integration Patterns

#### Batch Processing

```python
def batch_search(engine, patterns):
    """Process multiple search patterns efficiently."""
    results = []

    for pattern in patterns:
        query = Query(pattern=pattern, use_regex=True)
        result = engine.run(query)
        results.append({
            'pattern': pattern,
            'matches': len(result.items),
            'files': result.stats.files_matched,
            'time_ms': result.stats.elapsed_ms
        })

    return results

# Usage
patterns = ["def.*handler", "class.*Test", "import requests"]
batch_results = batch_search(engine, patterns)
```

#### Result Processing

```python
def analyze_results(results):
    """Analyze search results for insights."""

    # Group by file
    by_file = {}
    for item in results.items:
        if item.file not in by_file:
            by_file[item.file] = []
        by_file[item.file].append(item)

    # Find hotspots (files with many matches)
    hotspots = sorted(
        by_file.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:5]

    print("Top files with matches:")
    for file_path, items in hotspots:
        print(f"  {file_path}: {len(items)} matches")

    return by_file

# Usage
analysis = analyze_results(results)
```

---

## Search Types

### Text Search

Simple string matching within file contents.

```bash
# CLI
pysearch find --pattern "TODO"

# API
results = engine.search("TODO")
```

**Use cases:**

- Finding specific strings or comments
- Locating configuration values
- Searching for error messages

### Regex Search

Pattern matching using regular expressions.

```bash
# CLI
pysearch find --pattern "def.*handler" --regex

# API
results = engine.search(r"def.*handler", regex=True)
```

**Use cases:**

- Complex pattern matching
- Finding functions with specific naming patterns
- Matching structured data

### AST Search

Structural search using Abstract Syntax Tree parsing.

```bash
# CLI
pysearch find --pattern "def" --ast --filter-func-name "main"

# API
from pysearch.types import ASTFilters
filters = ASTFilters(func_name="main")
results = engine.search("def", use_ast=True, filters=filters)
```

**Use cases:**

- Finding specific code structures
- Locating functions, classes, or decorators
- Analyzing code patterns

### Semantic Search

Conceptual search using lightweight semantic analysis.

```bash
# CLI
pysearch find --pattern "database connection" --semantic

# API
results = engine.search("database connection", use_semantic=True)
```

**Use cases:**

- Finding conceptually related code
- Discovering similar functionality
- Code exploration and understanding

---

## Output Formats

### Text Format (Default)

Human-readable plain text output.

```bash
pysearch find --pattern "def main" --format text
```

**Features:**

- Easy to read
- Good for terminal output
- Supports context lines

### JSON Format

Machine-readable structured output.

```bash
pysearch find --pattern "def main" --format json
```

**Features:**

- Structured data
- Easy to parse programmatically
- Includes metadata and statistics

**Example output:**

```json
{
  "query": {
    "pattern": "def main",
    "use_regex": false
  },
  "stats": {
    "files_scanned": 42,
    "total_matches": 3,
    "elapsed_ms": 125.5
  },
  "items": [
    {
      "file": "src/main.py",
      "start_line": 10,
      "end_line": 12,
      "lines": ["def main():", "    print('Hello')", "    return 0"],
      "score": 1.0
    }
  ]
}
```

### Highlight Format

Interactive terminal output with syntax highlighting.

```bash
pysearch find --pattern "def main" --format highlight
```

**Features:**

- Syntax highlighting
- Color-coded matches
- Interactive terminal display
- Requires TTY support

---

## Advanced Features

### Caching

Enable result caching for improved performance on repeated searches.

```python
# Enable caching
engine.enable_caching(ttl=3600)  # 1 hour cache

# Custom cache directory
engine.enable_caching(cache_dir="./custom-cache", ttl=7200)
```

### File Watching

Automatically update search index when files change.

```python
# Enable auto-watching
engine.enable_auto_watch()

# Now the index updates automatically when files change
results = engine.search("pattern")  # Always uses fresh index
```

### Multi-Repository Search

Search across multiple repositories or projects.

```python
from pysearch.multi_repo import RepositoryInfo

# Configure repositories
repos = [
    RepositoryInfo(name="main", path="./", priority=1.0),
    RepositoryInfo(name="lib", path="../shared-lib", priority=0.8),
    RepositoryInfo(name="tools", path="../tools", priority=0.6)
]

# Enable multi-repo search
engine.enable_multi_repo(repos)

# Search across all repositories
results = engine.search("pattern")
```

### Custom Scoring

Implement custom result scoring and ranking.

```python
from pysearch.scorer import RankingStrategy

# Custom ranking configuration
config = SearchConfig(
    rank_strategy=RankingStrategy.DEFAULT,
    ast_weight=2.0,  # Boost AST matches
    text_weight=1.0  # Standard text matches
)
```

---

## Performance Tips

### Optimize Search Scope

```python
# Good: Specific paths and patterns
config = SearchConfig(
    paths=["./src"],  # Specific directory
    include=["**/*.py"],  # Specific file types
    exclude=["**/.venv/**", "**/__pycache__/**"]  # Exclude build artifacts
)

# Avoid: Too broad scope
config = SearchConfig(
    paths=["."],  # Entire project
    include=None,  # All files
    exclude=None  # No exclusions
)
```

### Enable Parallel Processing

```python
config = SearchConfig(
    parallel=True,
    workers=8,  # Adjust based on CPU cores
    strict_hash_check=False  # Faster file change detection
)
```

### Use Appropriate Search Types

```python
# Fast: Text search for simple patterns
results = engine.search("TODO")

# Medium: Regex for complex patterns
results = engine.search(r"def.*handler", regex=True)

# Slower: AST for structural search (but more precise)
results = engine.search("def", use_ast=True, filters=filters)
```

### Configure Content Filtering

```python
# Skip unnecessary content types for better performance
config = SearchConfig(
    enable_docstrings=False,  # Skip docstrings
    enable_comments=False,    # Skip comments
    enable_strings=True       # Keep string literals
)
```

### Set File Size Limits

```python
config = SearchConfig(
    file_size_limit=1_000_000,  # 1MB limit
    max_file_bytes=1_000_000    # Backup limit
)
```

---

## Troubleshooting

### Common Issues

#### No Matches Found

**Problem:** Search returns no results despite expecting matches.

**Solutions:**

1. Check include/exclude patterns:

   ```bash
   pysearch find --pattern "pattern" --include "**/*.py" --stats
   ```

2. Verify file paths:

   ```bash
   pysearch find --pattern "pattern" --path ./correct/path
   ```

3. Test with broader patterns:

   ```bash
   pysearch find --pattern "def" --regex  # Should find function definitions
   ```

#### Slow Performance

**Problem:** Search takes too long to complete.

**Solutions:**

1. Reduce search scope:

   ```bash
   pysearch find --pattern "pattern" --path ./specific/dir
   ```

2. Add exclusions:

   ```bash
   pysearch find --pattern "pattern" --exclude "**/.venv/**" "**/.git/**"
   ```

3. Disable unnecessary parsing:

   ```bash
   pysearch find --pattern "pattern" --no-docstrings --no-comments
   ```

4. Enable parallel processing:

   ```bash
   pysearch find --pattern "pattern" --parallel --workers 4
   ```

#### Encoding Issues

**Problem:** Files with special characters cause errors.

**Solutions:**

1. Ensure UTF-8 encoding:

   ```python
   # Check file encoding
   with open('file.py', 'rb') as f:
       raw = f.read()
       encoding = chardet.detect(raw)['encoding']
   ```

2. Configure encoding handling:

   ```python
   config = SearchConfig(
       # pysearch handles encoding automatically
       # but you can check file metadata
   )
   ```

#### Memory Issues

**Problem:** High memory usage on large codebases.

**Solutions:**

1. Set file size limits:

   ```python
   config = SearchConfig(file_size_limit=1_000_000)  # 1MB
   ```

2. Reduce parallel workers:

   ```python
   config = SearchConfig(workers=2)  # Fewer workers
   ```

3. Use streaming for large results:

   ```python
   # Process results in batches
   for item in results.items[:100]:  # First 100 results
       process_item(item)
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
from pysearch.logging_config import enable_debug_logging

enable_debug_logging()
results = engine.search("pattern")  # Will show detailed logs
```

```bash
# CLI debug mode
PYSEARCH_DEBUG=1 pysearch find --pattern "pattern"
```

### Getting Help

1. **Check documentation:** See [API Reference](api-reference.md) for detailed information
2. **Review examples:** Check the `examples/` directory for working code
3. **Run diagnostics:** Use `--stats` flag to see performance metrics
4. **Enable logging:** Use debug mode to see what's happening internally

### Performance Monitoring

```python
# Monitor search performance
results = engine.search("pattern")
stats = results.stats

print(f"Performance Report:")
print(f"  Files scanned: {stats.files_scanned}")
print(f"  Files matched: {stats.files_matched}")
print(f"  Total matches: {stats.total_matches}")
print(f"  Elapsed time: {stats.elapsed_ms:.1f}ms")
print(f"  Cache hits: {stats.cache_hits}")
print(f"  Cache misses: {stats.cache_misses}")

# Calculate efficiency metrics
if stats.files_scanned > 0:
    match_rate = stats.files_matched / stats.files_scanned
    print(f"  Match rate: {match_rate:.2%}")

if stats.cache_hits + stats.cache_misses > 0:
    cache_efficiency = stats.cache_hits / (stats.cache_hits + stats.cache_misses)
    print(f"  Cache efficiency: {cache_efficiency:.2%}")
```

---

## Next Steps

- **Advanced Configuration:** See [Configuration Guide](configuration.md)
- **API Reference:** Check [API Reference](api-reference.md) for detailed documentation
- **Architecture:** Learn about internals in [Architecture](architecture.md)
- **Examples:** Explore practical examples in the `examples/` directory
- **Contributing:** See [Contributing Guide](../CONTRIBUTING.md) to contribute
