# CLI Reference

Complete command-line interface reference for pysearch.

## Table of Contents

- [Overview](#overview)
- [Global Options](#global-options)
- [Commands](#commands)
- [Search Options](#search-options)
- [Output Options](#output-options)
- [Filter Options](#filter-options)
- [Performance Options](#performance-options)
- [Examples](#examples)
- [Exit Codes](#exit-codes)

---

## Overview

The pysearch command-line interface provides powerful search capabilities with extensive configuration options.

### Basic Syntax

```bash
pysearch [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Quick Examples

```bash
# Basic search
pysearch find --pattern "def main" --path ./src

# Regex search with context
pysearch find --pattern "def.*handler" --regex --context 5

# AST search with filters
pysearch find --pattern "def" --ast --filter-func-name "test_.*"

# JSON output for automation
pysearch find --pattern "TODO" --format json --output results.json
```

---

## Global Options

Options that apply to all commands.

### `--version`

Show version information and exit.

```bash
pysearch --version
```

### `--help`

Show help message and exit.

```bash
pysearch --help
pysearch find --help  # Command-specific help
```

### `--config FILE`

Use specific configuration file.

```bash
pysearch --config ./custom-config.toml find --pattern "pattern"
```

### `--verbose`, `-v`

Enable verbose output. Can be repeated for more verbosity.

```bash
pysearch -v find --pattern "pattern"      # Verbose
pysearch -vv find --pattern "pattern"     # Very verbose
pysearch -vvv find --pattern "pattern"    # Debug level
```

### `--quiet`, `-q`

Suppress non-essential output.

```bash
pysearch -q find --pattern "pattern"
```

---

## Commands

### `find`

Primary search command. Searches for patterns in files.

```bash
pysearch find [OPTIONS] --pattern PATTERN
```

**Required:**

- `--pattern PATTERN`: Search pattern (text or regex)

**Basic Options:**

- `--path PATH`: Search path (can be repeated)
- `--include PATTERN`: Include file pattern (can be repeated)
- `--exclude PATTERN`: Exclude file pattern (can be repeated)

### `config`

Configuration management commands.

```bash
# Show current configuration
pysearch config show

# Validate configuration file
pysearch config validate --file config.toml

# Generate example configuration
pysearch config example > pysearch.toml
```

### `cache`

Cache management commands.

```bash
# Show cache status
pysearch cache status

# Clear cache
pysearch cache clear

# Show cache statistics
pysearch cache stats
```

---

## Search Options

### Pattern Options

#### `--pattern PATTERN`

**Required.** The search pattern.

```bash
pysearch find --pattern "def main"
pysearch find --pattern "TODO|FIXME"  # With --regex
```

#### `--regex`, `-r`

Enable regular expression matching.

```bash
pysearch find --pattern "def \w+_handler" --regex
```

#### `--case-sensitive`

Enable case-sensitive matching (default: case-insensitive).

```bash
pysearch find --pattern "Class" --case-sensitive
```

### Search Type Options

#### `--ast`

Enable AST (Abstract Syntax Tree) search for code structure.

```bash
pysearch find --pattern "def" --ast --filter-func-name "test_.*"
```

#### `--semantic`

Enable semantic search for conceptual matching.

```bash
pysearch find --pattern "error handling" --semantic
```

### Scope Options

#### `--path PATH`

Search path. Can be specified multiple times.

```bash
pysearch find --pattern "pattern" --path ./src --path ./tests
```

#### `--include PATTERN`

Include files matching glob pattern. Can be repeated.

```bash
pysearch find --pattern "pattern" --include "**/*.py" --include "**/*.pyx"
```

#### `--exclude PATTERN`

Exclude files matching glob pattern. Can be repeated.

```bash
pysearch find --pattern "pattern" --exclude "**/.venv/**" --exclude "**/.git/**"
```

#### `--language LANG`

Limit search to specific programming languages.

```bash
pysearch find --pattern "pattern" --language python --language javascript
```

#### `--max-file-size SIZE`

Maximum file size to search (in bytes).

```bash
pysearch find --pattern "pattern" --max-file-size 1048576  # 1MB
```

---

## Output Options

### Format Options

#### `--format FORMAT`

Output format. Options: `text`, `json`, `highlight`.

```bash
pysearch find --pattern "pattern" --format json
pysearch find --pattern "pattern" --format highlight  # Syntax highlighting
```

#### `--output FILE`, `-o FILE`

Write output to file instead of stdout.

```bash
pysearch find --pattern "pattern" --format json --output results.json
```

### Context Options

#### `--context LINES`, `-C LINES`

Number of context lines around matches.

```bash
pysearch find --pattern "pattern" --context 5
```

#### `--before-context LINES`, `-B LINES`

Number of lines before matches.

```bash
pysearch find --pattern "pattern" --before-context 3
```

#### `--after-context LINES`, `-A LINES`

Number of lines after matches.

```bash
pysearch find --pattern "pattern" --after-context 3
```

### Limit Options

#### `--max-results COUNT`

Maximum number of results to return.

```bash
pysearch find --pattern "pattern" --max-results 100
```

#### `--max-files COUNT`

Maximum number of files to search.

```bash
pysearch find --pattern "pattern" --max-files 1000
```

---

## Filter Options

### AST Filters

Used with `--ast` flag to filter by code structure.

#### `--filter-func-name PATTERN`

Filter functions by name pattern (regex).

```bash
pysearch find --pattern "def" --ast --filter-func-name "test_.*"
```

#### `--filter-class-name PATTERN`

Filter classes by name pattern (regex).

```bash
pysearch find --pattern "class" --ast --filter-class-name ".*Manager"
```

#### `--filter-decorator PATTERN`

Filter by decorator pattern (regex).

```bash
pysearch find --pattern "@" --ast --filter-decorator "lru_cache"
```

#### `--filter-import PATTERN`

Filter by import pattern (regex).

```bash
pysearch find --pattern "import" --ast --filter-import "requests.*"
```

### Content Filters

#### `--no-docstrings`

Skip searching in docstrings.

```bash
pysearch find --pattern "pattern" --no-docstrings
```

#### `--no-comments`

Skip searching in comments.

```bash
pysearch find --pattern "pattern" --no-comments
```

#### `--no-strings`

Skip searching in string literals.

```bash
pysearch find --pattern "pattern" --no-strings
```

#### `--docstrings-only`

Search only in docstrings.

```bash
pysearch find --pattern "pattern" --docstrings-only
```

#### `--comments-only`

Search only in comments.

```bash
pysearch find --pattern "pattern" --comments-only
```

---

## Performance Options

### Parallel Processing

#### `--parallel`

Enable parallel processing (default: enabled).

```bash
pysearch find --pattern "pattern" --parallel
```

#### `--no-parallel`

Disable parallel processing.

```bash
pysearch find --pattern "pattern" --no-parallel
```

#### `--workers COUNT`

Number of worker threads (0 = auto-detect).

```bash
pysearch find --pattern "pattern" --workers 8
```

### Caching

#### `--cache`

Enable result caching.

```bash
pysearch find --pattern "pattern" --cache
```

#### `--no-cache`

Disable result caching.

```bash
pysearch find --pattern "pattern" --no-cache
```

#### `--cache-ttl SECONDS`

Cache time-to-live in seconds.

```bash
pysearch find --pattern "pattern" --cache --cache-ttl 3600  # 1 hour
```

### Memory Management

#### `--strict-hash-check`

Enable strict file change detection (slower but more accurate).

```bash
pysearch find --pattern "pattern" --strict-hash-check
```

#### `--no-dir-prune`

Disable directory pruning optimization.

```bash
pysearch find --pattern "pattern" --no-dir-prune
```

---

## Examples

### Basic Usage

```bash
# Simple text search
pysearch find --pattern "TODO" --path ./src

# Search in multiple directories
pysearch find --pattern "def main" --path ./src --path ./tests

# Search specific file types
pysearch find --pattern "import requests" --include "**/*.py"
```

### Advanced Searches

```bash
# Regex search for function patterns
pysearch find --pattern "def test_\w+" --regex --path ./tests

# AST search for async functions
pysearch find --pattern "async def" --ast --filter-func-name ".*handler"

# Semantic search for concepts
pysearch find --pattern "error handling" --semantic --path ./src
```

### Output Formatting

```bash
# JSON output for automation
pysearch find --pattern "TODO" --format json --output todos.json

# Highlighted output for terminal
pysearch find --pattern "class.*Test" --regex --format highlight

# Minimal output with no context
pysearch find --pattern "import" --context 0 --quiet
```

### Performance Optimization

```bash
# High-performance search
pysearch find --pattern "pattern" \
  --parallel --workers 8 \
  --cache --cache-ttl 3600 \
  --exclude "**/.venv/**" "**/.git/**" \
  --max-file-size 1048576

# Memory-efficient search
pysearch find --pattern "pattern" \
  --workers 2 \
  --context 1 \
  --max-results 100 \
  --no-docstrings --no-comments
```

### Complex Filtering

```bash
# Find test functions with specific decorators
pysearch find --pattern "def" --ast \
  --filter-func-name "test_.*" \
  --filter-decorator "pytest\.mark\.(parametrize|skip)" \
  --path ./tests

# Find classes with specific imports
pysearch find --pattern "class" --ast \
  --filter-class-name ".*Manager" \
  --filter-import "from.*import.*Manager" \
  --path ./src
```

---

## Exit Codes

pysearch uses standard exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success - matches found |
| 1 | No matches found |
| 2 | Error in command line arguments |
| 3 | File access error |
| 4 | Configuration error |
| 5 | Search error |
| 6 | Output error |

### Examples

```bash
# Check if pattern exists (exit code 0 = found, 1 = not found)
pysearch find --pattern "TODO" --quiet
echo $?  # 0 if found, 1 if not found

# Use in scripts
if pysearch find --pattern "deprecated" --quiet; then
    echo "Found deprecated code"
else
    echo "No deprecated code found"
fi
```

---

## Environment Variables

CLI options can be set via environment variables:

| Variable | Option | Example |
|----------|--------|---------|
| `PYSEARCH_PATHS` | `--path` | `export PYSEARCH_PATHS="./src:./tests"` |
| `PYSEARCH_INCLUDE` | `--include` | `export PYSEARCH_INCLUDE="**/*.py"` |
| `PYSEARCH_EXCLUDE` | `--exclude` | `export PYSEARCH_EXCLUDE="**/.venv/**"` |
| `PYSEARCH_CONTEXT` | `--context` | `export PYSEARCH_CONTEXT="5"` |
| `PYSEARCH_FORMAT` | `--format` | `export PYSEARCH_FORMAT="json"` |
| `PYSEARCH_PARALLEL` | `--parallel` | `export PYSEARCH_PARALLEL="true"` |
| `PYSEARCH_WORKERS` | `--workers` | `export PYSEARCH_WORKERS="8"` |
| `PYSEARCH_CACHE` | `--cache` | `export PYSEARCH_CACHE="true"` |

### Priority Order

Settings are applied in this order (later overrides earlier):

1. Default values
2. Configuration file
3. Environment variables
4. Command line arguments

---

## Configuration Files

CLI can use configuration files to set default options:

```toml
# pysearch.toml
[search]
paths = ["./src", "./tests"]
include = ["**/*.py"]
exclude = ["**/.venv/**", "**/.git/**"]
context = 3
parallel = true
workers = 4

[output]
format = "text"

[performance]
cache = true
cache_ttl = 3600
strict_hash_check = false
```

Use with:

```bash
pysearch --config pysearch.toml find --pattern "pattern"
```

---

## Shell Integration

### Bash Completion

Add to your `.bashrc`:

```bash
eval "$(pysearch completion bash)"
```

### Zsh Completion

Add to your `.zshrc`:

```bash
eval "$(pysearch completion zsh)"
```

### Fish Completion

```fish
pysearch completion fish | source
```

---

## See Also

- [Usage Guide](usage.md) - Comprehensive usage documentation
- [Configuration Guide](configuration.md) - Configuration options
- [API Reference](api-reference.md) - Python API documentation
- [Examples](../examples/README.md) - Practical examples
