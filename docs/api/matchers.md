# Matchers API

The matchers module provides core pattern matching functionality for different search types.

## Main Functions

### search_in_file

::: pysearch.search.matchers.search_in_file
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### find_text_regex_matches

::: pysearch.search.matchers.find_text_regex_matches
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### find_ast_blocks

::: pysearch.search.matchers.find_ast_blocks
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### ast_node_matches_filters

::: pysearch.search.matchers.ast_node_matches_filters
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Helper Functions

### group_matches_into_blocks

::: pysearch.search.matchers.group_matches_into_blocks
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Usage Examples

### Text Search

```python
from pysearch.matchers import search_in_file
from pysearch.types import Query
from pathlib import Path

# Simple text search
query = Query(pattern="def main")
results = search_in_file(
    Path("example.py"),
    "def main():\n    pass",
    query,
    context=2
)
```

### Regex Search

```python
# Regex search with named groups
query = Query(
    pattern=r"def (?P<name>\w+)\(",
    use_regex=True
)
results = search_in_file(path, content, query, context=3)
```

### AST Search with Filters

```python
from pysearch.types import ASTFilters

# Find functions with specific decorators
filters = ASTFilters(
    func_name=".*handler",
    decorator="lru_cache|cache"
)

query = Query(
    pattern="def",
    use_ast=True,
    ast_filters=filters
)

results = search_in_file(path, content, query, context=5)
```

### Semantic Search

```python
# Semantic search for conceptually related code
query = Query(
    pattern="database connection",
    use_semantic=True
)

results = search_in_file(path, content, query, context=3)
```

### Advanced AST Filtering

```python
# Complex AST filtering
filters = ASTFilters(
    func_name="test_.*",  # Test functions
    class_name=".*Test.*",  # Test classes
    decorator="pytest\\..*",  # Pytest decorators
    imported="requests\\.(get|post)"  # Specific imports
)

query = Query(
    pattern="",  # Empty pattern for AST-only search
    use_ast=True,
    ast_filters=filters
)
```

## Search Modes

### Text Mode
- Simple string matching
- Case-sensitive or case-insensitive
- Fastest search mode

### Regex Mode
- Full regex pattern support
- Named groups and backreferences
- Multiline mode support
- Enhanced regex engine with better Unicode support

### AST Mode
- Structure-aware code search
- Filter by function names, class names, decorators, imports
- Language-specific parsing
- Precise code element matching

### Semantic Mode
- Lightweight semantic matching
- Concept-based search using predefined patterns
- Finds conceptually related code even without exact matches
- No external models required

## Performance Tips

1. **Use specific patterns**: More specific patterns reduce false positives
2. **Combine modes**: Use AST filters with text/regex for precise results
3. **Limit context**: Smaller context windows improve performance
4. **Cache results**: The indexer caches file content for repeated searches

## Related

- [Types](types.md) - Query and result types
- [Semantic Search](semantic.md) - Advanced semantic features
- [Examples](../../examples/README.md) - More usage examples
