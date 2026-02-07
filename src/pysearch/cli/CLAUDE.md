# CLI Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **cli**

---

## Change Log (Changelog)

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive CLI module documentation

---

## Module Responsibility

The **CLI** module provides the command-line interface for PySearch:

1. **Command Parsing**: Command-line argument parsing and validation
2. **Interactive Interface**: User-friendly command-line interaction
3. **Output Formatting**: Terminal-optimized output formatting
4. **Help System**: Comprehensive help and documentation

---

## Key Files

| File | Purpose | Description |
|------|---------|-------------|
| `main.py` | Main CLI | Click-based command-line interface |
| `__init__.py` | Module Init | Module initialization and exports |
| `__main__.py` | Entry Point | Python module entry point |
| `README.md` | Module Docs | CLI module documentation |

---

## CLI Commands

### Primary Commands

#### `find` - Main Search Command

```bash
pysearch find [OPTIONS]
```

**Key Options**:
- `--pattern`: Search pattern (text or regex)
- `--path`: Search paths (default: current directory)
- `--include`: Include glob patterns
- `--exclude`: Exclude glob patterns
- `--regex`: Enable regex matching
- `--ast`: Enable AST-based matching
- `--semantic`: Enable semantic search
- `--context`: Number of context lines
- `--format`: Output format (text/json/highlight)
- `--filter-func-name`: Function name filter
- `--filter-class-name`: Class name filter
- `--filter-decorator`: Decorator filter
- `--filter-import`: Import filter
- `--rank`: Ranking strategy configuration

**Examples**:
```bash
# Basic text search
pysearch find --pattern "def main"

# Regex search with context
pysearch find --pattern "def.*handler" --regex --context 3

# AST-based search with filters
pysearch find --pattern "def" --ast --filter-func-name ".*handler"

# Semantic search
pysearch find --pattern "database connection" --semantic

# Boolean query
pysearch find --pattern "(async AND handler) NOT test" --logic

# Count-only search
pysearch find --pattern "TODO" --count
```

#### `index` - Indexing Management

```bash
pysearch index [OPTIONS] COMMAND
```

**Subcommands**:
- `build`: Build the search index
- `refresh`: Refresh the index
- `stats`: Show index statistics
- `clean`: Clean the index

#### `history` - Search History

```bash
pysearch history [OPTIONS] COMMAND
```

**Subcommands**:
- `list`: List search history
- `show`: Show a specific search
- `bookmark`: Bookmark a search
- `analytics`: Show search analytics

#### `config` - Configuration Management

```bash
pysearch config [OPTIONS] COMMAND
```

**Subcommands**:
- `show`: Show current configuration
- `set`: Set a configuration value
- `validate`: Validate configuration

---

## Entry Points

### CLI Entry Point
```python
# Entry point defined in pyproject.toml
[project.scripts]
pysearch = "pysearch.cli:main"

# Implementation
def main() -> None:
    """Main CLI entry point."""
    cli()
```

### Module Entry Point
```bash
python -m pysearch.cli
```

---

## CLI Architecture

### Command Groups
```python
@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """PySearch - High-performance code search engine."""
    pass

# Command groups
cli.add_command(find_cmd)
cli.add_command(index_cmd)
cli.add_command(history_cmd)
cli.add_command(config_cmd)
```

### Option Categories

#### Search Options
- Pattern matching modes (text, regex, AST, semantic)
- Context and output configuration
- Filtering options

#### Scope Options
- Path specification
- Include/exclude patterns
- Language filtering

#### Performance Options
- Parallel processing
- Worker count
- Cache configuration

#### Output Options
- Format selection
- Color/highlighting
- Statistics display

---

## Output Formats

### Text Format
Plain text output with basic formatting.

### JSON Format
Structured JSON output for programmatic use:
```bash
pysearch find --pattern "def main" --format json
```

### Highlight Format
Syntax-highlighted terminal output with Rich:
```bash
pysearch find --pattern "def main" --format highlight
```

---

## Configuration

### Command-Line Configuration
All options can be specified via command-line arguments:
```bash
pysearch find \
  --path ./src ./tests \
  --include "**/*.py" \
  --exclude "**/__pycache__/**" \
  --pattern "async def" \
  --regex \
  --context 5
```

### Environment Variables
Configuration via environment variables:
```bash
PYSEARCH_PATH=./src PYSEARCH_REGEX=true pysearch find "def main"
```

### Config File
Configuration via `pysearch_config.toml`:
```toml
[search]
paths = ["./src", "./tests"]
include = ["**/*.py"]
exclude = ["**/__pycache__/**"]
context = 3

[output]
format = "highlight"
```

---

## Dependencies

### Internal Dependencies
- `pysearch.core`: Configuration and types
- `pysearch.api`: Main search API
- `pysearch.utils`: Output formatting

### External Dependencies
- `click`: CLI framework
- `rich`: Terminal output formatting
- `pygments`: Syntax highlighting

---

## Testing

### Unit Tests
Located in `tests/unit/cli/`:
- `test_cli_comprehensive.py` - Comprehensive CLI tests
- `test_cli.py` - Basic CLI tests

### Integration Tests
Located in `tests/integration/`:
- `test_new_cli_features.py` - New CLI feature tests

---

## Common Usage Patterns

### Quick Search
```bash
# Search current directory
pysearch find "def main"

# Search specific paths
pysearch find --path ./src --pattern "class.*Test"
```

### Advanced Filtering
```bash
# AST filters
pysearch find \
  --pattern "def" \
  --ast \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache"

# Metadata filters
pysearch find \
  --pattern "TODO" \
  --filter-min-size "1KB" \
  --filter-languages "python"
```

### Output Options
```bash
# JSON output
pysearch find --pattern "def" --format json

# Highlighted output
pysearch find --pattern "def" --format highlight

# With statistics
pysearch find --pattern "def" --stats
```

### Performance Options
```bash
# Parallel processing
pysearch find --pattern "def" --parallel --workers 4

# Count-only (fast)
pysearch find --pattern "def" --count
```

---

## Related Files
- `README.md` - Module overview
- `docs/cli-reference.md` - CLI reference
- `docs/usage.md` - Usage examples
