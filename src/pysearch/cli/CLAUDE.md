# CLI Module

[根目录](../../../CLAUDE.md) > **cli**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Updated with all CLI commands documentation
- Enhanced command options documentation
- Synchronized with current CLI implementation

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive CLI module documentation

---

## Module Responsibility

The **CLI** module provides a comprehensive command-line interface for PySearch using the Click framework.

### Key Responsibilities
1. **Command-line Interface**: Primary user interface for PySearch
2. **Command Groups**: Organized commands for different features
3. **Argument Parsing**: Comprehensive option parsing with validation
4. **Output Formatting**: Multiple output formats (text, JSON, highlight)

---

## Entry and Startup

### Main Entry Point
- **`main.py`** - CLI implementation (2287 lines)
  - `main()` - CLI entry point
  - `find_cmd()` - Main search command
  - `semantic_cmd()` - Semantic search command
  - `history_cmd()` - History management
  - `index_cmd()` - Index management
  - And 10+ additional commands

---

## Public API

### CLI Commands

```bash
# Main search command
pysearch find --pattern "def main" --path . --regex --context 3

# Semantic search
pysearch semantic "database connection" --path . --threshold 0.7

# Index management
pysearch index --stats
pysearch index --rebuild

# History
pysearch history --limit 20 --analytics
pysearch history --frequent

# Bookmarks
pysearch bookmarks --add my-search --pattern "async def"
pysearch bookmarks --list

# Configuration
pysearch config --validate
pysearch config --format json

# Dependencies
pysearch deps --metrics
pysearch deps --circular

# Multi-repo
pysearch repo --enable
pysearch repo --add frontend ./frontend

# IDE integration
pysearch ide --definition file.py 10 my_function

# Distributed indexing
pysearch distributed --enable --workers 4

# Cache management
pysearch cache --stats
pysearch cache --clear

# Suggestions
pysearch suggest "conection"  # typo correction
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.core.api` - PySearch engine
- `pysearch.core.config` - SearchConfig
- `pysearch.core.types` - Query, ASTFilters, etc.
- `pysearch.utils.formatter` - Output formatting

### External Dependencies
- `click>=8.1.7` - CLI framework
- `rich>=13.7.1` - Terminal formatting

---

## Command Reference

### find - Main Search Command

**Options**:
- `--path` - Search paths (multiple allowed)
- `--include/--exclude` - Glob patterns
- `--regex` - Enable regex matching
- `--fuzzy` - Enable fuzzy search
- `--fuzzy-algorithm` - Algorithm choice
- `--context` - Context lines
- `--format` - Output format (text/json/highlight)
- `--filter-func-name` - AST filter: function name
- `--filter-class-name` - AST filter: class name
- `--filter-decorator` - AST filter: decorator
- `--filter-import` - AST filter: import
- `--stats` - Print statistics
- `--logic` - Boolean query mode
- `--count` - Count-only mode
- `--max-per-file` - Limit results per file
- `--multi-fuzzy` - Multi-algorithm fuzzy
- `--phonetic` - Phonetic search
- `--word-fuzzy` - Word-level fuzzy
- `--group-by-file` - Group by file

**Example**:
```bash
pysearch find "def .*_handler" \
  --regex \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache" \
  --format json \
  --context 3
```

### semantic - Semantic Search

**Options**:
- `--path` - Search paths
- `--threshold` - Similarity threshold (0.0-1.0)
- `--max-results` - Maximum results
- `--format` - Output format
- `--context` - Context lines

**Example**:
```bash
pysearch semantic "database connection" --threshold 0.7
```

### history - Search History

**Options**:
- `--limit` - Limit entries
- `--pattern` - Filter by pattern
- `--analytics` - Show analytics
- `--days` - Analytics days range
- `--sessions` - Show sessions
- `--tags` - Filter by tags
- `--clear` - Clear history
- `--frequent` - Show frequent patterns
- `--recent` - Show recent patterns
- `--rate` - Rate a search
- `--add-tags` - Add tags to search
- `--suggest` - Get suggestions
- `--end-session` - End current session
- `--performance-insights` - Show performance insights
- `--usage-patterns` - Analyze usage patterns
- `--session-analytics` - Session analytics
- `--cleanup-sessions` - Cleanup old sessions
- `--bookmark-search` - Search bookmarks

**Example**:
```bash
pysearch history --analytics --days 30
pysearch history --frequent --limit 10
```

### bookmarks - Bookmark Management

**Options**:
- `--add` - Add bookmark
- `--remove` - Remove bookmark
- `--pattern` - Search pattern (for adding)
- `--folder` - Folder name
- `--create-folder` - Create folder
- `--delete-folder` - Delete folder
- `--description` - Folder description
- `--list-folders` - List folders
- `--remove-from-folder` - Remove from folder

**Example**:
```bash
pysearch bookmarks --add my-search --pattern "async def"
pysearch bookmarks --create-folder auth --description "Auth-related searches"
```

### index - Index Management

**Options**:
- `--path` - Index paths
- `--stats` - Show statistics
- `--cleanup` - Cleanup old entries (days)
- `--rebuild` - Force rebuild

**Example**:
```bash
pysearch index --stats
pysearch index --cleanup 30
```

### deps - Dependency Analysis

**Options**:
- `--path` - Analysis path
- `--recursive/--no-recursive` - Recursive analysis
- `--metrics` - Show metrics
- `--impact` - Analyze impact
- `--suggest` - Suggest refactoring
- `--circular` - Detect circular dependencies
- `--coupling` - Show coupling metrics
- `--dead-code` - Detect dead code
- `--export` - Export graph (dot/json/csv)
- `--check-path` - Check dependency path
- `--format` - Output format (text/json)

**Example**:
```bash
pysearch deps --metrics
pysearch deps --circular --format json
```

### repo - Multi-Repository Search

**Options**:
- `--enable` - Enable multi-repo
- `--disable` - Disable multi-repo
- `--add` - Add repository (name path)
- `--priority` - Repository priority
- `--configure` - Configure repository
- `--remove` - Remove repository
- `--list` - List repositories
- `--info` - Repository info
- `--search` - Search across repos
- `--regex` - Use regex for search
- `--max-results` - Max results
- `--timeout` - Timeout per repo
- `--max-workers` - Max parallel workers
- `--health` - Health status
- `--stats` - Statistics
- `--format` - Output format

**Example**:
```bash
pysearch repo --enable --workers 4
pysearch repo --add frontend ./frontend --priority high
pysearch repo --search "async def" --max-results 1000
```

### ide - IDE Integration

**Options**:
- `--definition` - Jump to definition (file line symbol)
- `--references` - Find references (file line symbol)
- `--completion` - Auto-complete (file line col prefix)
- `--hover` - Hover info (file line col symbol)
- `--symbols` - List file symbols
- `--workspace-symbols` - Search workspace symbols
- `--diagnostics` - File diagnostics
- `--path` - Search paths
- `--format` - Output format

**Example**:
```bash
pysearch ide --definition main.py 10 my_function
pysearch ide --symbols main.py
```

### cache - Cache Management

**Options**:
- `--enable` - Enable cache (memory/disk)
- `--disable` - Disable cache
- `--clear` - Clear cache
- `--stats` - Show statistics
- `--cache-dir` - Cache directory
- `--max-size` - Max cache size
- `--ttl` - Cache TTL (seconds)
- `--hit-rate` - Show hit rate
- `--set-ttl` - Set TTL
- `--compression` - Enable compression
- `--invalidate-file` - Invalidate file

**Example**:
```bash
pysearch cache --enable memory --max-size 1000 --ttl 3600
pysearch cache --stats
```

---

## Testing

### Test Directory
- `tests/unit/cli/` - CLI tests
  - `test_main.py` - CLI command tests
  - `test_init.py` - CLI initialization tests

### Running Tests
```bash
pytest tests/unit/cli/ -v
```

---

## Common Issues and Solutions

### Issue 1: Command not found
**Symptoms**: `pysearch: command not found`
**Solution**: Ensure package is installed:
```bash
pip install -e .
```

### Issue 2: Options not working
**Symptoms**: CLI options ignored
**Solution**: Check for conflicting options:
```bash
# Can't use --fuzzy and --regex together
pysearch find --fuzzy --regex  # ERROR
```

### Issue 3: Output formatting issues
**Symptoms**: Malformed output
**Solution**: Check terminal compatibility:
```bash
# Use text format for non-interactive terminals
pysearch find --pattern "test" --format text
```

---

## Related Files

### CLI Module Files
- `src/pysearch/cli/__init__.py` - Package init
- `src/pysearch/cli/main.py` - Main CLI implementation
- `src/pysearch/cli/__main__.py` - Module entry point

---

## Module Structure

```
cli/
├── __init__.py          # Package initialization
├── __main__.py          # Module entry point (python -m pysearch)
└── main.py              # Click-based CLI implementation
```
