# Configuration Guide

This comprehensive guide covers all configuration options for pysearch, from basic setup to advanced performance tuning.

## Table of Contents

- [Overview](#overview)
- [Basic Configuration](#basic-configuration)
- [Search Scope Configuration](#search-scope-configuration)
- [Performance Configuration](#performance-configuration)
- [Content Filtering](#content-filtering)
- [Output Configuration](#output-configuration)
- [Advanced Options](#advanced-options)
- [Configuration Files](#configuration-files)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)

---

## Overview

pysearch uses the `SearchConfig` class for all configuration management. Configuration can be set through:

- **Python API**: Direct instantiation of `SearchConfig`
- **CLI parameters**: Command-line arguments
- **Configuration files**: TOML configuration files
- **Environment variables**: Environment-based settings

### Configuration Hierarchy

Settings are applied in this order (later overrides earlier):

1. Default values
2. Configuration file settings
3. Environment variables
4. CLI parameters
5. API parameters

---

## Basic Configuration

### Minimal Setup

```python
from pysearch import PySearch, SearchConfig

# Minimal configuration
config = SearchConfig()
engine = PySearch(config)

# Search current directory for Python files
results = engine.search("def main")
```

### Common Setup

```python
from pysearch.config import SearchConfig
from pysearch.types import OutputFormat

config = SearchConfig(
    paths=["./src", "./tests"],           # Search paths
    include=["**/*.py"],                  # Include patterns
    exclude=["**/.venv/**", "**/.git/**"], # Exclude patterns
    context=3,                            # Context lines
    output_format=OutputFormat.JSON,      # Output format
    parallel=True,                        # Enable parallel processing
    workers=4                             # Number of workers
)
```

---

## Search Scope Configuration

### Paths

Define which directories to search.

```python
config = SearchConfig(
    paths=[
        "./src",           # Source code
        "./tests",         # Test files
        "./docs",          # Documentation
        "../shared-lib"    # External library
    ]
)
```

**CLI equivalent:**

```bash
pysearch find --path ./src --path ./tests --path ./docs --pattern "pattern"
```

### Include Patterns

Specify which files to include using glob patterns.

```python
config = SearchConfig(
    include=[
        "**/*.py",         # Python files
        "**/*.pyx",        # Cython files
        "**/*.pyi",        # Type stub files
        "**/Dockerfile",   # Docker files
        "**/Makefile"      # Make files
    ]
)
```

**Auto-detection:** If `include` is `None`, pysearch automatically detects patterns based on supported languages.

### Exclude Patterns

Specify which files/directories to exclude.

```python
config = SearchConfig(
    exclude=[
        "**/.venv/**",         # Virtual environments
        "**/.git/**",          # Git directories
        "**/node_modules/**",  # Node.js modules
        "**/__pycache__/**",   # Python cache
        "**/build/**",         # Build artifacts
        "**/dist/**",          # Distribution files
        "**/.pytest_cache/**", # Pytest cache
        "**/htmlcov/**"        # Coverage reports
    ]
)
```

**Default exclusions:** If `exclude` is `None`, sensible defaults are applied automatically.

### Language Filtering

Limit search to specific programming languages.

```python
from pysearch.types import Language

config = SearchConfig(
    languages={
        Language.PYTHON,
        Language.JAVASCRIPT,
        Language.TYPESCRIPT
    }
)
```

**Supported languages:**

- Python (.py, .pyx, .pyi)
- JavaScript (.js, .jsx, .mjs)
- TypeScript (.ts, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- Rust (.rs)
- Go (.go)
- And more...

### File Size Limits

Control which files are processed based on size.

```python
config = SearchConfig(
    file_size_limit=2_000_000,  # 2MB limit
    max_file_bytes=2_000_000    # Backup limit (deprecated)
)
```

---

## Performance Configuration

### Parallel Processing

Enable multi-threaded search for better performance.

```python
config = SearchConfig(
    parallel=True,      # Enable parallel processing
    workers=8,          # Number of worker threads (0 = auto)
)
```

**Auto-detection:** Setting `workers=0` automatically uses `cpu_count()`.

### Caching Configuration

Configure file content and index caching.

```python
from pathlib import Path

config = SearchConfig(
    cache_dir=Path("./custom-cache"),  # Custom cache directory
    # Default: .pysearch-cache under first search path
)

# Enable caching in engine
engine = PySearch(config)
engine.enable_caching(ttl=3600)  # 1 hour cache
```

### Hash Verification

Control file change detection precision.

```python
config = SearchConfig(
    strict_hash_check=False  # Default: False for performance
)
```

**Options:**

- `True`: Compute SHA1 hash for precise change detection (slower)
- `False`: Use size/mtime only for change detection (faster)

### Directory Pruning

Optimize directory traversal by skipping excluded directories.

```python
config = SearchConfig(
    dir_prune_exclude=True  # Default: True
)
```

**Options:**

- `True`: Skip excluded directories during traversal (faster)
- `False`: Check all files individually (slower but same results)

---

## Content Filtering

### Content Type Toggles

Control which parts of files to search.

```python
config = SearchConfig(
    enable_docstrings=True,   # Search in docstrings
    enable_comments=True,     # Search in comments
    enable_strings=True       # Search in string literals
)
```

**Use cases:**

- Code-only search: `enable_docstrings=False, enable_comments=False`
- Documentation search: `enable_docstrings=True, enable_comments=False, enable_strings=False`
- Full-text search: All enabled (default)

### Symlink Handling

Control whether to follow symbolic links.

```python
config = SearchConfig(
    follow_symlinks=False  # Default: False for security
)
```

**Security note:** Following symlinks can lead to infinite loops or access to unintended files.

---

## Output Configuration

### Output Format

Choose the output format for results.

```python
from pysearch.types import OutputFormat

config = SearchConfig(
    output_format=OutputFormat.JSON  # JSON, TEXT, or HIGHLIGHT
)
```

**Formats:**

- `TEXT`: Human-readable plain text
- `JSON`: Machine-readable structured data
- `HIGHLIGHT`: Terminal with syntax highlighting

### Context Lines

Control how many lines of context to show around matches.

```python
config = SearchConfig(
    context=5  # Show 5 lines before and after each match
)
```

**Performance impact:** More context lines increase memory usage and output size.

### Ranking Configuration

Configure result scoring and ranking.

```python
from pysearch.config import RankStrategy

config = SearchConfig(
    rank_strategy=RankStrategy.DEFAULT,
    ast_weight=2.0,    # Boost AST matches
    text_weight=1.0    # Standard text matches
)
```

---

## Advanced Options

### Complete Configuration Example

```python
from pysearch.config import SearchConfig, RankStrategy
from pysearch.types import OutputFormat, Language
from pathlib import Path

config = SearchConfig(
    # Search scope
    paths=["./src", "./tests", "./docs"],
    include=["**/*.py", "**/*.pyx", "**/*.md"],
    exclude=[
        "**/.venv/**",
        "**/.git/**",
        "**/build/**",
        "**/__pycache__/**"
    ],
    languages={Language.PYTHON},

    # Behavior
    context=3,
    output_format=OutputFormat.JSON,
    follow_symlinks=False,
    file_size_limit=5_000_000,  # 5MB

    # Content filtering
    enable_docstrings=True,
    enable_comments=False,
    enable_strings=True,

    # Performance
    parallel=True,
    workers=6,
    cache_dir=Path("./custom-cache"),
    strict_hash_check=False,
    dir_prune_exclude=True,

    # Ranking
    rank_strategy=RankStrategy.DEFAULT,
    ast_weight=2.5,
    text_weight=1.0
)
```

### Dynamic Configuration

Modify configuration after creation:

```python
config = SearchConfig()

# Adjust for development
config.parallel = True
config.workers = 8
config.enable_comments = False

# Adjust for production
if production_mode:
    config.strict_hash_check = True
    config.file_size_limit = 1_000_000
```

---

## Configuration Files

### TOML Configuration

Create a `pysearch.toml` file:

```toml
# pysearch.toml
[search]
paths = ["./src", "./tests"]
include = ["**/*.py", "**/*.pyx"]
exclude = ["**/.venv/**", "**/.git/**", "**/__pycache__/**"]
context = 3
parallel = true
workers = 4

[content]
enable_docstrings = true
enable_comments = false
enable_strings = true

[performance]
strict_hash_check = false
dir_prune_exclude = true
file_size_limit = 2000000

[output]
format = "json"
```

### Loading Configuration Files

```python
import tomllib
from pysearch.config import SearchConfig

# Load from TOML file
with open("pysearch.toml", "rb") as f:
    config_data = tomllib.load(f)

# Create config from loaded data
config = SearchConfig(**config_data.get("search", {}))

# Apply other sections
if "content" in config_data:
    for key, value in config_data["content"].items():
        setattr(config, key, value)
```

### Configuration File Locations

pysearch looks for configuration files in this order:

1. `./pysearch.toml` (current directory)
2. `~/.config/pysearch/config.toml` (user config)
3. `/etc/pysearch/config.toml` (system config)

---

## Environment Variables

### Supported Variables

```bash
# Basic settings
export PYSEARCH_PATHS="./src:./tests"
export PYSEARCH_CONTEXT="5"
export PYSEARCH_PARALLEL="true"
export PYSEARCH_WORKERS="8"

# Content filtering
export PYSEARCH_ENABLE_DOCSTRINGS="false"
export PYSEARCH_ENABLE_COMMENTS="false"
export PYSEARCH_ENABLE_STRINGS="true"

# Performance
export PYSEARCH_STRICT_HASH_CHECK="false"
export PYSEARCH_DIR_PRUNE_EXCLUDE="true"
export PYSEARCH_FILE_SIZE_LIMIT="2000000"

# Output
export PYSEARCH_OUTPUT_FORMAT="json"

# Debug
export PYSEARCH_DEBUG="true"
export PYSEARCH_LOG_LEVEL="DEBUG"
```

### Loading Environment Variables

```python
import os
from pysearch.config import SearchConfig

config = SearchConfig()

# Override with environment variables
if "PYSEARCH_PARALLEL" in os.environ:
    config.parallel = os.environ["PYSEARCH_PARALLEL"].lower() == "true"

if "PYSEARCH_WORKERS" in os.environ:
    config.workers = int(os.environ["PYSEARCH_WORKERS"])

if "PYSEARCH_CONTEXT" in os.environ:
    config.context = int(os.environ["PYSEARCH_CONTEXT"])
```

---

## Best Practices

### Development Configuration

Optimized for fast iteration during development:

```python
dev_config = SearchConfig(
    paths=["./src"],
    exclude=["**/.venv/**", "**/.git/**", "**/__pycache__/**"],
    parallel=True,
    workers=4,
    strict_hash_check=False,  # Faster
    dir_prune_exclude=True,   # Skip excluded dirs
    enable_comments=False,    # Focus on code
    context=2                 # Minimal context
)
```

### Production Configuration

Optimized for accuracy and comprehensive results:

```python
prod_config = SearchConfig(
    paths=["./src", "./tests", "./docs"],
    parallel=True,
    workers=8,
    strict_hash_check=True,   # More accurate
    dir_prune_exclude=True,
    enable_docstrings=True,   # Include documentation
    enable_comments=True,     # Include comments
    context=5,                # More context
    file_size_limit=5_000_000 # Larger files
)
```

### CI/CD Configuration

Optimized for continuous integration:

```python
ci_config = SearchConfig(
    paths=["./src", "./tests"],
    parallel=True,
    workers=2,                # Limited resources
    strict_hash_check=True,   # Consistency
    dir_prune_exclude=True,
    file_size_limit=1_000_000, # Limit memory usage
    context=3
)
```

### Large Codebase Configuration

Optimized for very large repositories:

```python
large_config = SearchConfig(
    paths=["./src"],          # Limit scope
    exclude=[
        "**/.venv/**", "**/.git/**", "**/__pycache__/**",
        "**/node_modules/**", "**/build/**", "**/dist/**",
        "**/vendor/**", "**/third_party/**"
    ],
    parallel=True,
    workers=12,               # More workers
    strict_hash_check=False,  # Performance over precision
    dir_prune_exclude=True,   # Essential for large repos
    file_size_limit=500_000,  # Smaller limit
    enable_docstrings=False,  # Reduce processing
    context=1                 # Minimal context
)
```

### Configuration Validation

```python
def validate_config(config: SearchConfig) -> list[str]:
    """Validate configuration and return warnings."""
    warnings = []

    # Check for common issues
    if config.workers > 16:
        warnings.append("High worker count may cause resource contention")

    if config.file_size_limit > 10_000_000:
        warnings.append("Large file size limit may cause memory issues")

    if not config.exclude:
        warnings.append("No exclude patterns may slow down search")

    if config.context > 10:
        warnings.append("High context count increases output size")

    return warnings

# Usage
warnings = validate_config(config)
for warning in warnings:
    print(f"Warning: {warning}")
```

### Performance Monitoring

```python
def benchmark_config(config: SearchConfig, pattern: str) -> dict:
    """Benchmark a configuration with a test pattern."""
    import time

    engine = PySearch(config)

    start_time = time.time()
    results = engine.search(pattern)
    elapsed = time.time() - start_time

    return {
        "elapsed_seconds": elapsed,
        "files_scanned": results.stats.files_scanned,
        "matches_found": len(results.items),
        "files_per_second": results.stats.files_scanned / elapsed if elapsed > 0 else 0
    }

# Usage
benchmark = benchmark_config(config, "def main")
print(f"Performance: {benchmark['files_per_second']:.1f} files/second")
```

---

## Configuration Reference

### Complete Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `paths` | `list[str]` | `["."]` | Search paths |
| `include` | `list[str] \| None` | `None` | Include patterns (auto-detect if None) |
| `exclude` | `list[str] \| None` | `None` | Exclude patterns (defaults if None) |
| `languages` | `set[Language] \| None` | `None` | Language filter (auto-detect if None) |
| `file_size_limit` | `int` | `2_000_000` | File size limit in bytes |
| `context` | `int` | `2` | Context lines around matches |
| `output_format` | `OutputFormat` | `TEXT` | Output format |
| `follow_symlinks` | `bool` | `False` | Follow symbolic links |
| `enable_docstrings` | `bool` | `True` | Search in docstrings |
| `enable_comments` | `bool` | `True` | Search in comments |
| `enable_strings` | `bool` | `True` | Search in string literals |
| `parallel` | `bool` | `True` | Enable parallel processing |
| `workers` | `int` | `0` | Worker threads (0 = auto) |
| `cache_dir` | `Path \| None` | `None` | Cache directory |
| `strict_hash_check` | `bool` | `False` | Strict file change detection |
| `dir_prune_exclude` | `bool` | `True` | Prune excluded directories |
| `rank_strategy` | `RankStrategy` | `DEFAULT` | Ranking strategy |
| `ast_weight` | `float` | `2.0` | AST match weight |
| `text_weight` | `float` | `1.0` | Text match weight |

### Method Reference

| Method | Description |
|--------|-------------|
| `get_include_patterns()` | Get resolved include patterns |
| `get_exclude_patterns()` | Get resolved exclude patterns |
| `resolve_cache_dir()` | Get resolved cache directory |

---

## Troubleshooting Configuration

### Common Configuration Issues

1. **No matches found**
   - Check `include`/`exclude` patterns
   - Verify `paths` are correct
   - Ensure `languages` includes target files

2. **Slow performance**
   - Enable `parallel=True`
   - Set appropriate `workers` count
   - Use `dir_prune_exclude=True`
   - Add more `exclude` patterns

3. **High memory usage**
   - Reduce `file_size_limit`
   - Lower `workers` count
   - Decrease `context` lines

4. **Missing results**
   - Check content toggles (`enable_docstrings`, etc.)
   - Verify `file_size_limit` isn't too restrictive
   - Ensure `follow_symlinks` setting is appropriate

### Debug Configuration

```python
from pysearch.logging_config import enable_debug_logging

# Enable debug logging
enable_debug_logging()

# Create config with debug info
config = SearchConfig(paths=["./src"])
print(f"Include patterns: {config.get_include_patterns()}")
print(f"Exclude patterns: {config.get_exclude_patterns()}")
print(f"Cache directory: {config.resolve_cache_dir()}")
```

---

## See Also

- [Usage Guide](usage.md) - How to use pysearch effectively
- [API Reference](api-reference.md) - Complete API documentation
- [Architecture](architecture.md) - Internal design and components
