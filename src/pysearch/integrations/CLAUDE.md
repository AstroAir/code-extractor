# Integrations Module

[根目录](../../../CLAUDE.md) > **integrations**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Added distributed indexing documentation
- Enhanced multi-repo search documentation
- Updated IDE integration documentation
- Synchronized with current project structure

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive integrations module documentation

---

## Module Responsibility

The **Integrations** module provides external system integrations including multi-repository search, distributed indexing, and IDE hooks.

### Key Responsibilities
1. **Multi-Repository Search**: Search across multiple code repositories
2. **Distributed Indexing**: Parallel distributed indexing for large codebases
3. **IDE Integration**: LSP-like features (jump-to-definition, find references)

---

## Entry and Startup

### Main Entry Points
- **`multi_repo.py`** - Multi-repository search
  - `MultiRepoSearchEngine` - Multi-repo search engine
  - `add_repository()` - Add repository
  - `search_all()` - Search across all repos

- **`distributed_indexing.py`** - Distributed indexing
  - `DistributedIndexingEngine` - Distributed indexing engine
  - `index_codebase()` - Index with multiple workers
  - `scale_workers()` - Scale worker count

- **`ide_hooks.py`** - IDE integration
  - `IDEIntegration` - IDE feature integration
  - `jump_to_definition()` - Jump to definition
  - `find_references()` - Find references

---

## Public API

### Multi-Repository Search

```python
from pysearch.integrations import MultiRepoSearchEngine

engine = MultiRepoSearchEngine(max_workers=4)

# Add repositories
engine.add_repository("frontend", "./frontend", priority="high")
engine.add_repository("backend", "./backend", priority="normal")

# Search across all repositories
results = engine.search_all_repositories(
    pattern="async def",
    use_regex=True,
    max_results=1000
)

# Get repository info
info = engine.get_repository_info("frontend")
print(f"Files: {info.file_count}")
```

### Distributed Indexing

```python
from pysearch.integrations import DistributedIndexingEngine
import asyncio

engine = DistributedIndexingEngine(num_workers=4)

async def index_codebase():
    updates = await engine.index_codebase(
        directories=["./src", "./tests"],
        batch_size=100
    )
    for update in updates:
        print(f"{update['status']}: {update['description']}")

asyncio.run(index_codebase())

# Scale workers
success = await engine.scale_distributed_workers(8)
```

### IDE Integration

```python
from pysearch.integrations import IDEIntegration

ide = IDEIntegration(config)

# Jump to definition
result = ide.jump_to_definition(
    file_path="main.py",
    line=10,
    symbol="my_function"
)

# Find references
refs = ide.find_references(
    file_path="main.py",
    line=10,
    symbol="my_function"
)

# Get document symbols
symbols = ide.get_document_symbols("main.py")

# Provide completions
completions = ide.provide_completion(
    file_path="main.py",
    line=15,
    column=10,
    prefix="my_"
)

# Get diagnostics
diags = ide.get_diagnostics("main.py")
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.core.api` - PySearch engine
- `pysearch.core.config` - Configuration
- `pysearch.indexing` - Indexing system

### External Dependencies
- No special external dependencies
- Uses Python standard library for async/parallel processing

---

## Data Models

### Multi-Repo Types
- `RepositoryInfo` - Repository metadata
- `RepositoryHealth` - Repository health status
- `MultiRepoSearchResult` - Multi-repo search results

### Distributed Indexing Types
- `WorkerStats` - Worker statistics
- `ProgressUpdate` - Indexing progress
- `PerformanceMetrics` - Performance metrics

### IDE Types
- `SymbolInfo` - Symbol information
- `CompletionItem` - Completion item
- `Diagnostic` - Diagnostic information

---

## Testing

### Test Directory
- `tests/unit/integrations/` - Integrations tests
  - `test_multi_repo.py` - Multi-repo tests
  - `test_distributed_indexing.py` - Distributed indexing tests
  - `test_ide_hooks.py` - IDE integration tests
- `tests/integration/` - Integration tests
  - `test_multi_repo.py` - Multi-repo integration tests
  - `test_ide_hooks.py` - IDE hooks integration tests

### Running Tests
```bash
pytest tests/unit/integrations/ -v
pytest tests/integration/test_multi_repo.py -v
pytest tests/integration/test_ide_hooks.py -v
```

---

## Common Issues and Solutions

### Issue 1: Multi-repo search slow
**Symptoms**: Multi-repo search takes too long
**Solution**: Adjust timeout and worker count:
```python
engine = MultiRepoSearchEngine(
    max_workers=8,
    default_timeout=60.0
)
```

### Issue 2: Distributed indexing memory issues
**Symptoms**: High memory usage during indexing
**Solution**: Reduce batch size or worker count:
```python
engine = DistributedIndexingEngine(
    num_workers=2,
    batch_size=50
)
```

### Issue 3: IDE integration not working
**Symptoms**: IDE features return no results
**Solution**: Ensure indexing is complete:
```python
# Index first
engine.index_codebase(directories=["./src"])
# Then use IDE features
```

---

## Related Files

### Integrations Module Files
- `src/pysearch/integrations/__init__.py`
- `src/pysearch/integrations/multi_repo.py` - Multi-repo search
- `src/pysearch/integrations/distributed_indexing.py` - Distributed indexing
- `src/pysearch/integrations/ide_hooks.py` - IDE integration

---

## Module Structure

```
integrations/
├── __init__.py
├── multi_repo.py           # Multi-repository search
├── distributed_indexing.py # Distributed indexing
└── ide_hooks.py            # IDE integration hooks
```
