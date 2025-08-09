# Types API

Core data types and structures used throughout the PySearch system.

## Core Types

### Query

::: pysearch.types.Query
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### SearchResult

::: pysearch.types.SearchResult
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### SearchItem

::: pysearch.types.SearchItem
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### SearchStats

::: pysearch.types.SearchStats
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Enumerations

### OutputFormat

::: pysearch.types.OutputFormat
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Language

::: pysearch.types.Language
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Filter Types

### ASTFilters

::: pysearch.types.ASTFilters
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### MetadataFilters

::: pysearch.types.MetadataFilters
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Metadata Types

### FileMetadata

::: pysearch.types.FileMetadata
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Type Examples

### Creating Queries

```python
from pysearch.types import Query, ASTFilters, OutputFormat

# Simple text query
query = Query(pattern="def main")

# Regex query
query = Query(pattern=r"def\s+\w+_handler", use_regex=True)

# AST query with filters
filters = ASTFilters(func_name=".*handler", decorator="lru_cache")
query = Query(pattern="def", use_ast=True, ast_filters=filters)

# Semantic query
query = Query(pattern="database connection", use_semantic=True)
```

### Working with Results

```python
from pysearch.types import SearchResult, SearchItem

def process_results(results: SearchResult) -> None:
    print(f"Search completed in {results.stats.elapsed_ms}ms")
    print(f"Found {results.stats.items} matches in {results.stats.files_matched} files")
    
    for item in results.items:
        print(f"\n{item.file}:{item.start_line}-{item.end_line}")
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            print(f"{line_num:4d} | {line}")
```

### Advanced Filtering

```python
from pysearch.types import MetadataFilters
from datetime import datetime, timedelta

# Filter by file metadata
filters = MetadataFilters(
    min_size=1024,  # At least 1KB
    max_size=1024*1024,  # At most 1MB
    modified_after=datetime.now() - timedelta(days=7),  # Modified in last week
    min_lines=10,  # At least 10 lines
    authors={"alice", "bob"}  # Only files by specific authors
)
```

## Related

- [PySearch API](pysearch.md) - Main API interface
- [Configuration](config.md) - Configuration types
- [Examples](../../examples/README.md) - Usage examples
