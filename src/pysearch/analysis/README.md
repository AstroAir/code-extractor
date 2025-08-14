# Analysis Module

The analysis module provides code analysis and understanding capabilities.

## Responsibilities

- **Dependency Analysis**: Code dependency tracking and graph generation
- **Language Detection**: Programming language identification and support
- **Content Analysis**: Code content addressing and hashing
- **GraphRAG**: Graph-based retrieval-augmented generation (in `graphrag/` subdirectory)

## Key Files

- `dependency_analysis.py` - Dependency graph generation and analysis
- `language_detection.py` - Programming language detection
- `content_addressing.py` - Content hashing and addressing
- `language_support.py` - Enhanced language-specific processing (renamed from `enhanced_language_support.py`)
- `graphrag/` - GraphRAG functionality for code understanding

## Analysis Features

1. **Dependency Graphs**: Visual representation of code dependencies
2. **Circular Dependency Detection**: Identify problematic dependency cycles
3. **Language Support**: Multi-language code analysis
4. **Content Addressing**: Efficient content identification and deduplication

## Usage

```python
from pysearch.analysis import DependencyAnalyzer, detect_language

analyzer = DependencyAnalyzer()
graph = analyzer.analyze_directory("./src")
language = detect_language("example.py")
```
