# Search Module

[根目录](../../../CLAUDE.md) > **search**

---

## Change Log (Changelog)

### 2026-02-08 - Module Documentation Update
- Updated with fuzzy search algorithms
- Enhanced semantic search documentation
- Added boolean query system documentation
- Synchronized with current project structure

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive search module documentation

---

## Module Responsibility

The **Search** module provides pattern matching algorithms including text, regex, AST-based, fuzzy, semantic, and boolean search capabilities.

### Key Responsibilities
1. **Pattern Matching**: Text and regex matching
2. **AST Search**: Abstract Syntax Tree-based structural search
3. **Fuzzy Search**: Approximate string matching with multiple algorithms
4. **Semantic Search**: Concept-based search with optional transformers
5. **Boolean Search**: Logical query composition (AND, OR, NOT)
6. **Result Scoring**: Ranking and deduplication

---

## Entry and Startup

### Main Entry Points
- **`matchers.py`** - Pattern matching implementation
  - `search_in_file()` - Main search function
  - `text_search()` - Text pattern matching
  - `regex_search()` - Regex pattern matching
  - `ast_search()` - AST-based search

- **`fuzzy.py`** - Fuzzy search algorithms
  - `fuzzy_search()` - Fuzzy matching
  - Multi-algorithm support (Levenshtein, Damerau-Levenshtein, Jaro-Winkler, etc.)

- **`boolean.py`** - Boolean query system
  - `BooleanQueryParser` - Parse boolean queries
  - `BooleanQueryEvaluator` - Evaluate boolean queries

- **`scorer.py`** - Result scoring and ranking
  - `rank_results()` - Rank search results
  - `deduplicate_results()` - Remove duplicates

---

## Public API

### Basic Search

```python
from pysearch.search import search_in_file
from pysearch.core.types import Query

query = Query(pattern="def main", use_regex=True)
results = search_in_file(file_path, content, query, context=3)
```

### Fuzzy Search

```python
from pysearch.search.fuzzy import fuzzy_search_advanced, FuzzyAlgorithm

results = fuzzy_search_advanced(
    query="authetication",  # typo intended
    candidates=candidates,
    algorithm=FuzzyAlgorithm.LEVENSHTEIN,
    threshold=0.7
)
```

### Boolean Search

```python
from pysearch.search.boolean import BooleanQueryParser, BooleanQueryEvaluator

parser = BooleanQueryParser()
evaluator = BooleanQueryEvaluator()

query = "(async AND handler) NOT test"
parsed = parser.parse(query)
results = evaluator.evaluate(parsed, search_items)
```

### Semantic Search

```python
from pysearch.search.semantic import semantic_search

results = semantic_search(
    query="database connection",
    file_contents=contents,
    threshold=0.7
)
```

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch.core.types` - Query, SearchItem, ASTFilters
- `pysearch.utils.helpers` - Context extraction, AST utilities

### External Dependencies
- `regex` - Enhanced regex engine
- `rapidfuzz>=3.0.0` - Fast fuzzy string matching
- `fuzzywuzzy>=0.18.0` - Fuzzy matching algorithms
- `python-levenshtein>=0.21.0` - Levenshtein distance

### Optional Dependencies
- `[semantic]` - Advanced semantic search
  - `sentence-transformers>=2.2.0`
  - `transformers>=4.30.0`
  - `torch>=2.0.0`

---

## Data Models

### Match Types
- `TextMatch` - Text match with line/column positions
- `ASTMatch` - AST-based match with node information

### Search Types
- `FuzzyAlgorithm` - Fuzzy algorithm enum
- `FuzzyMatch` - Fuzzy match result
- `SemanticMatch` - Semantic match result

### Boolean Types
- `BooleanOperator` - AND, OR, NOT operators
- `BooleanExpression` - Parsed boolean expression

---

## Testing

### Test Directory
- `tests/unit/search/` - Search module tests
  - `test_matchers.py` - Pattern matching tests
  - `test_fuzzy.py` - Fuzzy search tests
  - `test_boolean.py` - Boolean query tests
  - `test_semantic.py` - Semantic search tests
  - `test_scorer.py` - Scoring tests

### Running Tests
```bash
pytest tests/unit/search/ -v
pytest tests/unit/search/test_fuzzy.py -v
pytest tests/unit/search/test_boolean.py -v
```

---

## Common Issues and Solutions

### Issue 1: Regex too slow
**Symptoms**: Regex search hangs
**Solution**: Simplify pattern or use text search:
```python
# Instead of complex regex
query = Query(pattern=".*complex.*", use_regex=False)
```

### Issue 2: Fuzzy search low quality
**Symptoms**: Poor fuzzy match results
**Solution**: Adjust threshold or try different algorithm:
```python
results = fuzzy_search_advanced(
    query,
    candidates,
    algorithm=FuzzyAlgorithm.JARO_WINKLER,
    threshold=0.8  # Increase for stricter matching
)
```

### Issue 3: AST search fails
**Symptoms**: AST parsing errors
**Solution**: Ensure file is valid Python code or skip non-Python files

---

## Related Files

### Search Module Files
- `src/pysearch/search/__init__.py`
- `src/pysearch/search/matchers.py` - Pattern matching
- `src/pysearch/search/fuzzy.py` - Fuzzy search
- `src/pysearch/search/boolean.py` - Boolean queries
- `src/pysearch/search/semantic.py` - Lightweight semantic search
- `src/pysearch/search/semantic_advanced.py` - Advanced semantic search
- `src/pysearch/search/scorer.py` - Result scoring

---

## Module Structure

```
search/
├── __init__.py
├── matchers.py              # Pattern matching (text, regex, AST)
├── fuzzy.py                 # Fuzzy search algorithms
├── boolean.py               # Boolean query system
├── semantic.py              # Lightweight semantic search
├── semantic_advanced.py     # Advanced semantic search (transformers)
└── scorer.py                # Result scoring and ranking
```
