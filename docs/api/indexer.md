# Indexer API

The indexer provides efficient file scanning and caching capabilities with incremental updates.

## Indexer

::: pysearch.indexer.Indexer
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

## IndexRecord

::: pysearch.indexer.IndexRecord
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

### Basic Indexing

```python
from pysearch.indexer import Indexer
from pysearch.config import SearchConfig

config = SearchConfig(
    paths=["./src"],
    include=["**/*.py"],
    exclude=["**/__pycache__/**"]
)

indexer = Indexer(config)

# Get all files
files = list(indexer.iter_files())
print(f"Found {len(files)} files")

# Check if file needs update
needs_update = indexer.needs_update("src/example.py")
```

### Performance Monitoring

```python
import time

start_time = time.time()
files = list(indexer.iter_files())
elapsed = time.time() - start_time

print(f"Indexed {len(files)} files in {elapsed:.2f}s")

# Get cache statistics
cache_stats = indexer.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
```

### Strict Hash Checking

```python
# Enable strict hash checking for exact change detection
config = SearchConfig(
    paths=["."],
    strict_hash_check=True
)

indexer = Indexer(config)

# This will compute SHA1 hashes for all files
files = list(indexer.iter_files())
```

### Cache Management

```python
# Clear cache
indexer.clear_cache()

# Save cache manually
indexer.save_cache()

# Get cache directory
cache_dir = indexer.cache_dir
print(f"Cache stored in: {cache_dir}")
```

## Performance Considerations

### Incremental Updates

The indexer uses file metadata (size, mtime) to detect changes:

- **Fast mode** (default): Uses size and modification time
- **Strict mode**: Additionally computes SHA1 hashes for exact change detection

### Directory Pruning

When `dir_prune_exclude=True`, the indexer skips excluded directories entirely during traversal, improving performance for large codebases with many excluded paths.

### Parallel Scanning

The indexer supports parallel file scanning when `parallel=True` in the configuration.

## Cache Format

The cache is stored as JSON with the following structure:

```json
{
  "version": 1,
  "files": {
    "relative/path/to/file.py": {
      "path": "relative/path/to/file.py",
      "size": 1234,
      "mtime": 1640995200.0,
      "sha1": "abc123...",
      "last_accessed": 1640995200.0,
      "access_count": 5
    }
  }
}
```

## Related

- [Configuration](config.md) - Indexer configuration options
- [Performance](../performance.md) - Performance tuning guide
- [Utils](utils.md) - File utilities
