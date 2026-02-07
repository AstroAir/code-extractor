# PySearch API

The main PySearch class provides the primary interface for programmatic access to the search engine.

::: pysearch.api.PySearch
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      group_by_category: true
      show_bases: true
      show_inheritance_diagram: false
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google
      merge_init_into_class: true

## Usage Examples

### Basic Search

```python
from pysearch import PySearch, SearchConfig

# Create configuration
config = SearchConfig(
    paths=["."],
    include=["**/*.py"],
    context=3
)

# Initialize search engine
engine = PySearch(config)

# Perform search
results = engine.search("def main")
print(f"Found {len(results.items)} matches")
```

### Advanced Query

```python
from pysearch.types import Query, ASTFilters

# Create advanced query with AST filters
filters = ASTFilters(
    func_name=".*handler",
    decorator="lru_cache"
)

query = Query(
    pattern="def",
    use_ast=True,
    filters=filters,
    context=5
)

results = engine.run(query)
```

### Multi-Repository Search

```python
from pysearch.integrations.multi_repo import MultiRepoSearchEngine

# Search across multiple repositories
multi_engine = MultiRepoSearchEngine()
multi_engine.add_repository("project1", "/path/to/project1")
multi_engine.add_repository("project2", "/path/to/project2")

results = multi_engine.search_all("async def")
```

## Related

- [Configuration](config.md) - Search configuration options
- [Types](types.md) - Data types and structures
- [Examples](../../examples/README.md) - More usage examples
