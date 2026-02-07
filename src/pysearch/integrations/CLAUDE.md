# Integrations Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **integrations**

---

## Change Log (Changelog)

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive Integrations module documentation

---

## Module Responsibility

The **Integrations** module handles external system integrations and third-party connections:

1. **Multi-Repository**: Search across multiple repositories
2. **Distributed Processing**: Distributed indexing and search
3. **IDE Integration**: Integration with development environments
4. **External Services**: Third-party service connections

---

## Key Files

| File | Purpose | Description |
|------|---------|-------------|
| `multi_repo.py` | Multi-Repository | Multi-repository search capabilities |
| `distributed_indexing.py` | Distributed | Distributed indexing across nodes |
| `ide_hooks.py` | IDE Hooks | IDE integration and hooks |
| `types.py` | Integration Types | Integration-specific type definitions |
| `__init__.py` | Module Init | Module initialization and exports |
| `README.md` | Module Docs | Integrations module documentation |

---

## Multi-Repository Search

### Overview
Provides unified search across multiple code repositories with intelligent coordination.

### Key Classes
```python
class MultiRepoSearchEngine:
    def __init__(self, max_workers: int = 4)
    def add_repository(
        self,
        name: str,
        path: Path | str,
        config: SearchConfig | None = None,
        priority: str = "normal",
        **metadata
    ) -> bool
    def remove_repository(self, name: str) -> bool
    def search_all(
        self,
        pattern: str,
        use_regex: bool = False,
        use_ast: bool = False,
        use_semantic: bool = False,
        **kwargs
    ) -> MultiRepoSearchResult | None
    def search_repositories(
        self,
        repositories: list[str],
        query: Query,
        **kwargs
    ) -> MultiRepoSearchResult | None
    def get_health_status(self) -> dict[str, Any]
    def get_search_statistics(self) -> dict[str, Any]

class RepositoryInfo:
    name: str
    path: Path
    config: SearchConfig | None
    priority: str
    metadata: dict[str, Any]
    enabled: bool
```

### Multi-Repository Result
```python
class MultiRepoSearchResult:
    repository_results: dict[str, SearchResult]
    successful_repositories: int
    failed_repositories: int
    total_items: int
    aggregated_result: SearchResult | None
    errors: dict[str, str]
```

### Usage
```python
from pysearch.integrations import MultiRepoSearchEngine, RepositoryInfo

# Initialize engine
engine = MultiRepoSearchEngine(max_workers=4)

# Add repositories
engine.add_repository("frontend", "./frontend", priority="high")
engine.add_repository("backend", "./backend")
engine.add_repository("shared", "./shared-lib")

# Search across all repositories
results = engine.search_all("async def", use_regex=True)

# Search specific repositories
results = engine.search_repositories(
    ["frontend", "backend"],
    query
)
```

---

## Distributed Indexing

### Overview
Provides distributed indexing capabilities for scaling across multiple machines.

### Key Classes
```python
class DistributedIndexingEngine:
    def __init__(self, coordinator_address: str)
    async def distribute_indexing(
        self,
        repositories: list[RepositoryInfo],
        chunk_size: int = 1000
    ) -> DistributedIndexingResult
    async def aggregate_results(
        self,
        node_results: list[IndexingResult]
    ) -> AggregatedResult
    def register_node(self, node_address: str) -> bool
    def get_node_status(self, node_address: str) -> NodeStatus | None
```

### Features
- **Horizontal Scaling**: Distribute indexing across multiple nodes
- **Fault Tolerance**: Handle node failures gracefully
- **Load Balancing**: Distribute work efficiently
- **Result Aggregation**: Combine results from multiple nodes

---

## IDE Integration

### Overview
Provides hooks and integration points for IDE and editor plugins.

### Key Classes
```python
class IDEHooks:
    def register_jump_to_definition(self, handler: Callable) -> str
    def register_find_references(self, handler: Callable) -> str
    def register_search_handler(self, handler: Callable) -> str
    def trigger_hook(self, hook_id: str, **kwargs) -> Any

class IDEIntegration:
    def __init__(self, ide_type: str)
    def provide_completion(
        self,
        context: str,
        position: tuple[int, int]
    ) -> list[CompletionItem]
    def provide_definition(
        self,
        file_path: str,
        position: tuple[int, int]
    ) -> DefinitionLocation | None
    def provide_references(
        self,
        file_path: str,
        position: tuple[int, int]
    ) -> list[ReferenceLocation]
```

### Supported IDEs
- **VS Code**: Via language server protocol
- **JetBrains**: Via plugin API
- **Vim/Neovim**: Via plugin
- **Emacs**: Via package

### Usage
```python
from pysearch.integrations import IDEIntegration, IDEHooks

# Create IDE integration
integration = IDEIntegration(ide_type="vscode")

# Register hooks
hooks = IDEHooks()
hook_id = hooks.register_search_handler(
    lambda query: engine.search(query.pattern)
)

# Provide completions
completions = integration.provide_completion(
    context="def my_fun",
    position=(0, 10)
)
```

---

## Integration Types

### Repository Types
```python
class RepositoryType(str, Enum):
    GIT = "git"
    MERCURIAL = "mercurial"
    SVN = "svn"
    LOCAL = "local"
```

### Priority Levels
```python
class Priority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
```

### Node Status
```python
class NodeStatus:
    address: str
    status: str  # "active", "inactive", "error"
    load: float
    last_heartbeat: float
    repositories_processed: int
```

---

## Dependencies

### Internal Dependencies
- `pysearch.core`: Configuration and API
- `pysearch.indexing`: Indexing functionality
- `pysearch.search`: Search algorithms

### External Dependencies
- `aiohttp`: Async HTTP client (for distributed)
- `watchdog`: File watching (for IDE sync)

---

## Testing

### Integration Tests
Located in `tests/integration/`:
- `test_multi_repo_*.py` - Multi-repository tests
- `test_ide_hooks.py` - IDE integration tests

---

## Common Usage Patterns

### Multi-Repository Search
```python
from pysearch import PySearch, SearchConfig
from pysearch.integrations import MultiRepoSearchEngine

# Enable multi-repo
config = SearchConfig(paths=["."])
engine = PySearch(config)
engine.enable_multi_repo(max_workers=4)

# Add repositories
engine.add_repository("project-a", "/path/to/project-a")
engine.add_repository("project-b", "/path/to/project-b")

# Search across all
results = engine.search_all_repositories("def main")

# Search specific repos
from pysearch.types import Query
query = Query(pattern="async def")
results = engine.search_specific_repositories(
    ["project-a"],
    query
)
```

### Health Monitoring
```python
# Get health status
health = engine.get_multi_repo_health()

for repo_name, status in health.items():
    print(f"{repo_name}: {status['status']}")

# Get search statistics
stats = engine.get_multi_repo_stats()
print(f"Total searches: {stats['total_searches']}")
```

---

## Configuration

### Repository Configuration
```python
from pysearch import SearchConfig

# Custom config per repository
config = SearchConfig(
    paths=["./src"],
    include=["**/*.py"],
    exclude=["**/tests/**"]
)

# Add with custom config
engine.add_repository(
    "my-repo",
    "/path/to/repo",
    config=config,
    priority="high",
    description="My main repository"
)
```

### Distributed Configuration
```python
from pysearch.integrations import DistributedIndexingEngine

# Configure distributed indexing
engine = DistributedIndexingEngine(
    coordinator_address="coordinator.example.com:8080"
)

# Register nodes
engine.register_node("node1.example.com:8080")
engine.register_node("node2.example.com:8080")
```

---

## Performance Considerations

### Multi-Repository
- Use parallel search for better performance
- Configure `max_workers` based on available cores
- Consider repository priorities for search ordering

### Distributed Indexing
- Choose appropriate `chunk_size` for data distribution
- Monitor node health and performance
- Implement proper error handling for network failures

### IDE Integration
- Cache search results for better performance
- Use async operations for non-blocking UI
- Implement proper error handling for file access

---

## Related Files
- `README.md` - Module overview
- `docs/architecture.md` - Architecture details
- `docs/api/multi_repo.md` - Multi-repo API reference
