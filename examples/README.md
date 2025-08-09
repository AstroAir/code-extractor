# PySearch Examples

This directory contains comprehensive examples demonstrating various features and use cases of PySearch. Each example is self-contained and includes detailed explanations and comments.

## Quick Start

To run any example:

```bash
# From the project root directory
python examples/basic_usage.py
python examples/advanced_examples.py
# ... etc
```

## Example Organization

### 🎓 [Tutorials](tutorials/) - Step-by-Step Learning

**Perfect for learning pysearch systematically:**

- **[Getting Started](tutorials/01_getting_started.py)** - Your first search with pysearch
- **[Search Types](tutorials/05_search_types.py)** - Text, regex, AST, and semantic search
- **[AST Filtering](tutorials/06_ast_filtering.py)** - Advanced code structure filtering
- **[Performance Optimization](tutorials/07_performance_optimization.py)** - Making searches faster
- **[Custom Workflows](tutorials/09_custom_workflows.py)** - Building search-based tools

[📖 **See all tutorials →**](tutorials/README.md)

## Example Files

### 📚 Core Examples

#### `basic_usage.py`
**Fundamental PySearch operations**
- Basic text search
- Regex pattern matching
- AST-based structural search
- Configuration and setup
- Result processing and formatting

**Key concepts covered:**
- SearchConfig setup
- PySearch engine initialization
- Query construction
- Result interpretation

#### `advanced_examples.py`
**Advanced search capabilities**
- Semantic search for conceptual matching
- Metadata filtering (file size, date, author)
- Custom scoring and ranking strategies
- Error handling and logging
- Performance optimization techniques

**Key concepts covered:**
- Semantic search with embeddings
- MetadataFilters usage
- Custom ranking strategies
- Debug logging setup

### 🔧 Specialized Features

#### `cache_and_watch.py`
**Caching and file watching**
- Cache configuration and management
- File watching for automatic updates
- Performance optimization with caching
- Cache invalidation strategies

**Key concepts covered:**
- CacheManager configuration
- FileWatcher setup
- Auto-indexing on file changes
- Cache performance tuning

#### `mcp_server_example.py`
**Model Context Protocol (MCP) integration**
- MCP server setup and configuration
- Exposing PySearch via MCP tools
- Advanced search operations through MCP
- Error handling in MCP context

**Key concepts covered:**
- MCP server initialization
- Tool registration
- Async search operations
- Response formatting

### 🎯 Practical Applications

#### `use_cases.py`
**Real-world usage scenarios**
- Code refactoring assistance
- Documentation generation
- Code quality analysis
- Dependency tracking
- Architecture analysis

**Key concepts covered:**
- Multi-repository search
- Dependency analysis
- Code metrics extraction
- Refactoring suggestions

#### `edge_cases.py`
**Handling complex scenarios**
- Large file handling
- Binary file detection
- Encoding issues
- Performance edge cases
- Error recovery

**Key concepts covered:**
- Robust error handling
- Performance monitoring
- Memory management
- Graceful degradation

### 🖥️ Command Line Interface

#### `cli_examples.sh`
**Command-line usage examples**
- Basic CLI operations
- Advanced filtering options
- Output formatting
- Batch processing
- Integration with other tools

**Key concepts covered:**
- CLI argument patterns
- Output redirection
- Scripting integration
- Performance tuning flags

## Common Patterns

### Basic Search Setup

```python
from pysearch import PySearch, SearchConfig

# Configure search parameters
config = SearchConfig(
    paths=["./src"],
    include=["**/*.py"],
    exclude=["**/test_*", "**/__pycache__/**"],
    context=3,
    parallel=True
)

# Initialize search engine
engine = PySearch(config)

# Perform search
results = engine.search("def main", regex=False)
```

### Advanced Query Construction

```python
from pysearch.types import Query, ASTFilters, MetadataFilters

# AST-based search with filters
ast_filters = ASTFilters(
    func_name=".*handler",
    decorator="lru_cache",
    class_name=".*Manager"
)

metadata_filters = MetadataFilters(
    max_file_size=1024*1024,  # 1MB
    languages=[Language.PYTHON],
    modified_after="2024-01-01"
)

query = Query(
    pattern="async def",
    use_regex=True,
    use_ast=True,
    ast_filters=ast_filters,
    metadata_filters=metadata_filters,
    context=5
)

results = engine.run(query)
```

### Result Processing

```python
# Process search results
for item in results.items:
    print(f"File: {item.file}")
    print(f"Lines {item.start_line}-{item.end_line}:")
    
    for i, line in enumerate(item.lines):
        line_num = item.start_line + i
        print(f"  {line_num:4d}: {line}")
    
    # Access match spans for highlighting
    for span in item.match_spans:
        print(f"  Match at line {span.line}, cols {span.start}-{span.end}")
```

## Performance Tips

1. **Use appropriate include/exclude patterns** to limit search scope
2. **Enable parallel processing** for large codebases
3. **Configure caching** for repeated searches
4. **Use AST filters** to narrow structural searches
5. **Set reasonable context limits** to avoid excessive output

## Error Handling

All examples include proper error handling patterns:

```python
try:
    results = engine.search(pattern)
    if results.items:
        # Process results
        pass
    else:
        print("No matches found")
except SearchError as e:
    print(f"Search error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Integration Examples

### With IDEs
- VS Code extension integration
- Vim/Neovim plugin usage
- Emacs integration patterns

### With CI/CD
- Code quality checks
- Documentation validation
- Dependency analysis

### With Other Tools
- Git hooks integration
- Pre-commit checks
- Code review automation

## Contributing

When adding new examples:

1. Follow the existing code style and documentation patterns
2. Include comprehensive docstrings and comments
3. Add error handling and edge case coverage
4. Update this README with the new example
5. Test examples with various Python versions

## Support

For questions about these examples or PySearch usage:

- Check the main documentation in `docs/`
- Review the API documentation
- Open an issue on the project repository
- Join the community discussions
