# Integrations Module

The integrations module handles external system integrations and third-party connections.

## Responsibilities

- **Multi-Repository**: Search across multiple repositories
- **Distributed Processing**: Distributed indexing and search
- **IDE Integration**: Integration with development environments
- **External Services**: Third-party service connections

## Key Files

- `multi_repo.py` - Multi-repository search capabilities
- `distributed_indexing.py` - Distributed indexing across multiple nodes
- `ide_hooks.py` - IDE integration and hooks

## Integration Features

1. **Multi-Repo Search**: Unified search across multiple code repositories
2. **Distributed Indexing**: Scale indexing across multiple machines
3. **IDE Hooks**: Integration with popular IDEs and editors
4. **Service APIs**: RESTful APIs for external service integration

## Usage

```python
from pysearch.integrations import MultiRepoSearchEngine, RepositoryInfo

repos = [
    RepositoryInfo(name="repo1", path="/path/to/repo1"),
    RepositoryInfo(name="repo2", path="/path/to/repo2")
]
engine = MultiRepoSearchEngine(repos)
results = engine.search("def main")
```
