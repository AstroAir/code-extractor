# Core Module

The core module contains the fundamental components of the pysearch package that other modules depend on.

## Responsibilities

- **Main API**: Primary search engine interface (`PySearch` class)
- **Configuration**: Search configuration management (`SearchConfig`)
- **Data Types**: Core data structures and type definitions
- **History**: Search history tracking and management

## Key Files

- `api.py` - Main PySearch class and primary API
- `config.py` - Configuration management and validation
- `types.py` - Core data types, enums, and structures
- `history.py` - Search history tracking

## Dependencies

This module has minimal external dependencies and serves as the foundation for other modules. It should not import from other pysearch modules except for basic utilities.

## Usage

```python
from pysearch.core import PySearch, SearchConfig
from pysearch.core.types import Query, SearchResult

config = SearchConfig(paths=["."], include=["**/*.py"])
engine = PySearch(config)
results = engine.search("def main")
```
