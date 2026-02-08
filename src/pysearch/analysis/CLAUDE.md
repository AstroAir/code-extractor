# Analysis Module

[根目录](../../../CLAUDE.md) > **analysis**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Enhanced GraphRAG documentation
- Added language support documentation
- Updated dependency analysis documentation
- Synchronized with current project structure

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive analysis module documentation

---

## Module Responsibility

The **Analysis** module provides code analysis capabilities including language detection, dependency analysis, content addressing, and GraphRAG (Graph Retrieval-Augmented Generation) for intelligent code understanding.

### Key Responsibilities
1. **Language Detection**: Automatic programming language identification (20+ languages)
2. **Dependency Analysis**: Import graph construction and circular dependency detection
3. **Content Addressing**: SHA256-based content deduplication
4. **GraphRAG**: Knowledge graph construction for enhanced code search

---

## Entry and Startup

### Main Entry Points
- **`language_detection.py`** - Language detection
  - `detect_language()` - Detect programming language from file
  - `get_supported_languages()` - List supported languages
  - `get_language_extensions()` - Get file extensions for language

- **`dependency_analysis.py`** - Dependency analysis
  - `DependencyAnalyzer` - Analyze code dependencies
  - `build_dependency_graph()` - Build dependency graph
  - `find_cycles()` - Detect circular dependencies

- **`graphrag/engine.py`** - GraphRAG engine
  - `GraphRAGEngine` - Knowledge graph construction
  - `extract_entities()` - Extract code entities
  - `build_knowledge_graph()` - Build knowledge graph
  - `graphrag_search()` - Graph-enhanced search

---

## Public API

### Language Detection

```python
from pysearch.analysis import detect_language, get_supported_languages

# Detect language from file path
language = detect_language(file_path)

# Get all supported languages
languages = get_supported_languages()
print(languages)  # [PYTHON, JAVASCRIPT, TYPESCRIPT, ...]
```

### Dependency Analysis

```python
from pysearch.analysis import DependencyAnalyzer

analyzer = DependencyAnalyzer(root_path="./src")
graph = analyzer.build_dependency_graph()

# Find circular dependencies
cycles = analyzer.find_cycles(graph)
for cycle in cycles:
    print(f"Cycle: {' -> '.join(cycle)}")

# Calculate metrics
metrics = analyzer.calculate_metrics()
print(f"Coupling: {metrics.avg_coupling}")
```

### GraphRAG

```python
from pysearch.analysis.graphrag import GraphRAGEngine
from pysearch import PySearch, SearchConfig

config = SearchConfig(paths=["./src"], enable_graphrag=True)
engine = PySearch(config)

# Build knowledge graph
engine.build_knowledge_graph()

# GraphRAG search
from pysearch.core.types import GraphRAGQuery
query = GraphRAGQuery(
    query="functions related to authentication",
    entity_types=["function", "class"],
    max_depth=3
)
results = engine.graphrag_search(query)
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.core.types` - GraphRAG types (CodeEntity, KnowledgeGraph, etc.)
- `pysearch.storage` - Vector database for graph storage

### External Dependencies
- No special dependencies for basic analysis
- GraphRAG requires: `qdrant-client>=1.7.0`, `numpy>=1.24.0`
- Optional: `sentence-transformers>=2.2.0` for embeddings

---

## Data Models

### Language Detection
- `Language` - Programming language enum (26 languages)

### Dependency Analysis
- `DependencyGraph` - Directed graph of dependencies
- `DependencyMetrics` - Coupling and cohesion metrics

### GraphRAG Types (`graphrag_types.py`)
- `CodeEntity` - Code entity (function, class, variable)
- `EntityType` - Entity type enum
- `EntityRelationship` - Relationship between entities
- `RelationType` - Relation type enum (IMPORTS, CALLS, INHERITS)
- `KnowledgeGraph` - Knowledge graph structure
- `GraphRAGQuery` - GraphRAG query specification
- `GraphRAGResult` - GraphRAG search result

---

## Testing

### Test Directory
- `tests/unit/analysis/` - Analysis module tests
  - `test_language_detection.py` - Language detection tests
  - `test_dependency_analysis.py` - Dependency analysis tests
  - `test_content_addressing.py` - Content addressing tests
  - `test_language_support.py` - Language support tests
  - `graphrag/` - GraphRAG tests

### Running Tests
```bash
pytest tests/unit/analysis/ -v
pytest tests/unit/analysis/graphrag/ -v
pytest tests/integration/test_graphrag.py -v
```

---

## Common Issues and Solutions

### Issue 1: Language detection inaccurate
**Symptoms**: Wrong language detected
**Solution**: Use file extension hints:
```python
language = detect_language(file_path, hint=Language.PYTHON)
```

### Issue 2: Dependency graph too large
**Symptoms**: Memory issues with large codebases
**Solution**: Analyze specific directories:
```python
analyzer = DependencyAnalyzer(root_path="./src", max_depth=3)
```

### Issue 3: GraphRAG slow on first run
**Symptoms**: Initial graph building is slow
**Solution**: Enable caching:
```python
config = SearchConfig(
    paths=["./src"],
    enable_graphrag=True,
    cache_dir="./.pysearch-cache"
)
```

---

## Related Files

### Analysis Module Files
- `src/pysearch/analysis/__init__.py`
- `src/pysearch/analysis/dependency_analysis.py` - Dependency analysis
- `src/pysearch/analysis/language_detection.py` - Language detection
- `src/pysearch/analysis/language_support.py` - Multi-language support
- `src/pysearch/analysis/content_addressing.py` - Content addressing
- `src/pysearch/analysis/graphrag/` - GraphRAG implementation (3 files)

---

## Module Structure

```
analysis/
├── __init__.py
├── dependency_analysis.py   # Dependency graph analysis
├── language_detection.py    # Programming language detection
├── language_support.py      # Multi-language processing support
├── content_addressing.py    # SHA256-based content addressing
└── graphrag/                # GraphRAG implementation
    ├── __init__.py
    ├── core.py              # GraphRAG core
    └── engine.py            # GraphRAG engine
```
