# Configuration API

The configuration system provides comprehensive control over search behavior, performance, and output formatting.

## SearchConfig

::: pysearch.config.SearchConfig
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

## RankStrategy

::: pysearch.config.RankStrategy
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Configuration Examples

### Basic Configuration

```python
from pysearch import SearchConfig, OutputFormat

config = SearchConfig(
    paths=["./src", "./tests"],
    include=["**/*.py"],
    exclude=["**/__pycache__/**", "**/.venv/**"],
    context=3,
    output_format=OutputFormat.JSON
)
```

### Performance-Optimized Configuration

```python
config = SearchConfig(
    paths=["./large_codebase"],
    parallel=True,
    workers=8,
    strict_hash_check=False,
    dir_prune_exclude=True,
    file_size_limit=5_000_000  # 5MB limit
)
```

### Language-Specific Configuration

```python
from pysearch import Language

config = SearchConfig(
    paths=["."],
    languages={Language.PYTHON, Language.JAVASCRIPT},
    enable_docstrings=True,
    enable_comments=True,
    enable_strings=False
)
```

### Advanced Ranking Configuration

```python
config = SearchConfig(
    paths=["."],
    rank_strategy=RankStrategy.DEFAULT,
    ast_weight=2.5,
    text_weight=1.0
)
```

## Environment Variables

The following environment variables can be used to override configuration:

- `PYSEARCH_CACHE_DIR`: Override default cache directory
- `PYSEARCH_WORKERS`: Set number of parallel workers
- `PYSEARCH_MAX_FILE_SIZE`: Set maximum file size limit
- `PYSEARCH_DEBUG`: Enable debug logging

## Configuration Files

PySearch supports TOML configuration files:

```toml
# pysearch.toml
[search]
paths = ["./src", "./tests"]
include = ["**/*.py", "**/*.js"]
exclude = ["**/__pycache__/**"]
context = 3
parallel = true
workers = 4

[output]
format = "json"
highlight = true

[performance]
file_size_limit = 2000000
strict_hash_check = false
```

## Related

- [Types](types.md) - Configuration-related types
- [Performance](../guide/performance.md) - Performance tuning guide
- [Configuration Guide](../guide/configuration.md) - Detailed configuration documentation
