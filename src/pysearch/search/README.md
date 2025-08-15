# Search Module

The search module implements various search strategies and pattern matching algorithms.

## Responsibilities

- **Pattern Matching**: Text, regex, and AST-based pattern matching
- **Fuzzy Search**: Approximate string matching capabilities
- **Semantic Search**: Basic and advanced semantic similarity search
- **Result Scoring**: Ranking and scoring of search results

## Key Files

- `matchers.py` - Core pattern matching implementations
- `fuzzy.py` - Fuzzy search algorithms
- `semantic.py` - Basic semantic search functionality
- `semantic_advanced.py` - Advanced semantic search features
- `scorer.py` - Result ranking and scoring algorithms

## Search Strategies

1. **Text Search**: Exact string matching
2. **Regex Search**: Regular expression pattern matching
3. **AST Search**: Abstract syntax tree-based structural matching
4. **Fuzzy Search**: Approximate matching with configurable similarity thresholds
5. **Semantic Search**: Vector-based semantic similarity matching

## Usage

```python
from pysearch.search import search_in_file, semantic_similarity_score
from pysearch.core.types import Query

# Text search
results = search_in_file(file_path, Query(pattern="def main"))

# Semantic similarity
score = semantic_similarity_score("function definition", "def main")
```
