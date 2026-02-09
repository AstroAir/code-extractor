# Multi-Repository API

Multi-repository search capabilities for searching across multiple codebases simultaneously.

## MultiRepoSearchEngine

::: pysearch.integrations.multi_repo.MultiRepoSearchEngine
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      group_by_category: true
      show_bases: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## MultiRepoSearchResult

::: pysearch.integrations.multi_repo.MultiRepoSearchResult
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## RepositoryInfo

::: pysearch.integrations.multi_repo.RepositoryInfo
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      group_by_category: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Usage Examples

### Basic Multi-Repository Search

```python
from pysearch.integrations.multi_repo import MultiRepoSearchEngine

# Initialize multi-repo engine
engine = MultiRepoSearchEngine()

# Add repositories
engine.add_repository("frontend", "/path/to/frontend")
engine.add_repository("backend", "/path/to/backend")
engine.add_repository("shared", "/path/to/shared-lib")

# Search across all repositories
results = engine.search_all("async def")
print(f"Found matches in {len(results.repository_results)} repositories")

for repo_name, repo_results in results.repository_results.items():
    print(f"{repo_name}: {len(repo_results.items)} matches")
```

### Repository Management

```python
# List all repositories
repos = engine.list_repositories()
for repo_info in repos:
    print(f"{repo_info.name}: {repo_info.path} ({repo_info.status})")

# Remove repository
engine.remove_repository("old-repo")

# Update repository path
engine.update_repository("frontend", "/new/path/to/frontend")
```

### Targeted Repository Search

```python
# Search specific repositories only
results = engine.search_repositories(
    pattern="def handler",
    repository_names=["backend", "shared"],
    use_regex=True
)

# Search with different configurations per repository
configs = {
    "frontend": SearchConfig(include=["**/*.js", "**/*.ts"]),
    "backend": SearchConfig(include=["**/*.py"]),
}

results = engine.search_with_configs("TODO", configs)
```

### Advanced Multi-Repository Operations

```python
from pysearch import Query, ASTFilters

# Complex query across repositories
filters = ASTFilters(func_name=".*handler", decorator="route")
query = Query(
    pattern="def",
    use_ast=True,
    filters=filters,
    context=5
)

results = engine.run_query_all(query)

# Aggregate statistics
total_files = sum(r.stats.files_scanned for r in results.repository_results.values())
total_matches = sum(len(r.items) for r in results.repository_results.values())

print(f"Scanned {total_files} files, found {total_matches} matches")
```

### Repository Health Monitoring

```python
# Check repository health
health = engine.check_repository_health("backend")
print(f"Repository health: {health.status}")
print(f"Issues: {health.issues}")

# Get repository statistics
stats = engine.get_repository_stats("backend")
print(f"Files: {stats.file_count}")
print(f"Size: {stats.total_size_mb:.1f} MB")
print(f"Languages: {stats.languages}")
```

## Configuration

### Repository-Specific Configuration

```python
from pysearch import SearchConfig, Language

# Configure each repository differently
engine.configure_repository("frontend", SearchConfig(
    include=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
    languages={Language.JAVASCRIPT, Language.TYPESCRIPT},
    context=2
))

engine.configure_repository("backend", SearchConfig(
    include=["**/*.py"],
    languages={Language.PYTHON},
    context=5,
    enable_docstrings=True
))
```

### Global Configuration

```python
# Set global defaults for all repositories
engine.set_global_config(SearchConfig(
    parallel=True,
    workers=4,
    file_size_limit=5_000_000
))
```

## Performance Optimization

### Parallel Repository Search

```python
# Enable parallel search across repositories
engine.enable_parallel_search(max_workers=8)

# Search repositories in parallel
results = engine.search_all_parallel("def main")
```

### Caching and Indexing

```python
# Enable cross-repository caching
engine.enable_shared_cache("/shared/cache/directory")

# Preindex all repositories
engine.preindex_all_repositories()

# Incremental updates
engine.update_indexes()
```

### Memory Management

```python
# Configure memory limits for large multi-repo searches
engine.configure_memory_limits(
    max_memory_mb=2048,
    max_results_per_repo=1000
)
```

## Integration Patterns

### CI/CD Integration

```python
def search_across_microservices(pattern):
    """Search pattern across all microservices."""
    engine = MultiRepoSearchEngine()
    
    # Auto-discover repositories from CI environment
    for service in os.environ.get("MICROSERVICES", "").split(","):
        if service.strip():
            path = f"/workspace/{service}"
            if Path(path).exists():
                engine.add_repository(service, path)
    
    return engine.search_all(pattern)
```

### Development Workflow

```python
def find_similar_implementations(code_snippet):
    """Find similar code implementations across repositories."""
    engine = MultiRepoSearchEngine()
    
    # Add all project repositories
    engine.add_repository("web-app", "./web-app")
    engine.add_repository("mobile-app", "./mobile-app")
    engine.add_repository("api-server", "./api-server")
    
    # Use semantic search to find similar implementations
    query = Query(pattern=code_snippet, use_semantic=True)
    results = engine.run_query_all(query)
    
    return results
```

## Error Handling

### Repository Errors

```python
try:
    results = engine.search_all("pattern")
except MultiRepoError as e:
    print(f"Multi-repo search failed: {e}")
    
    # Handle individual repository errors
    for repo_name, error in e.repository_errors.items():
        print(f"Error in {repo_name}: {error}")
```

### Graceful Degradation

```python
# Search with error tolerance
results = engine.search_all_tolerant("pattern", ignore_errors=True)

# Check which repositories had errors
for repo_name, repo_result in results.repository_results.items():
    if repo_result.error:
        print(f"Warning: {repo_name} search failed: {repo_result.error}")
```

## Related

- [PySearch API](pysearch.md) - Single repository search
- [Configuration](config.md) - Repository configuration
- [Types](types.md) - Multi-repository result types
- [Performance](../guide/performance.md) - Performance optimization
