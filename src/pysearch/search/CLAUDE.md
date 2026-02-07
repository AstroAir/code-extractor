# Search Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **search**

---

## Change Log (Changelog)

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive module documentation
- Documented matchers, fuzzy search, semantic search, and scoring

---

## Module Responsibility

The **Search** module implements various search strategies and pattern matching algorithms:

1. **Pattern Matching**: Text, regex, and AST-based pattern matching
2. **Fuzzy Search**: Approximate string matching with multiple algorithms
3. **Semantic Search**: Basic and advanced semantic similarity search
4. **Result Scoring**: Ranking and scoring of search results
5. **Boolean Logic**: Logical query operators (AND, OR, NOT)

---

## Key Files

| File | Purpose | Description |
|------|---------|-------------|
| `matchers.py` | Pattern Matching | Core pattern matching implementations |
| `fuzzy.py` | Fuzzy Search | Fuzzy search algorithms and utilities |
| `semantic.py` | Basic Semantic | Lightweight semantic search |
| `semantic_advanced.py` | Advanced Semantic | Embedding-based semantic search |
| `scorer.py` | Result Scoring | Ranking and scoring algorithms |
| `boolean.py` | Boolean Logic | Boolean query evaluation |

---

## Pattern Matching (matchers.py)

### Overview
The `matchers.py` module provides core pattern matching implementations for text, regex, AST, and semantic search.

### Key Functions
```python
def search_in_file(
    path: Path,
    content: str,
    query: Query
) -> list[SearchItem]
```

### Search Types

#### Text Search
Exact string matching with context extraction.

#### Regex Search
Enhanced regex matching using the `regex` library with support for:
- Multiline patterns
- Named groups
- Unicode properties

#### AST Search
Abstract syntax tree-based structural matching supporting:
- Function definition matching
- Class definition matching
- Decorator filtering
- Import statement filtering

#### Semantic Search
Lightweight semantic matching using:
- Token-based similarity
- Identifier matching
- Code structure awareness

---

## Fuzzy Search (fuzzy.py)

### Overview
The `fuzzy.py` module provides fuzzy matching algorithms for approximate string matching.

### Supported Algorithms
- **Levenshtein**: Standard edit distance
- **Damerau-Levenshtein**: Edit distance with transpositions
- **Jaro-Winkler**: String similarity for short strings
- **Soundex**: Phonetic encoding
- **Metaphone**: Phonetic encoding (improved)

### Key Functions
```python
def fuzzy_pattern(
    pattern: str,
    max_distance: int = 2,
    algorithm: FuzzyAlgorithm = FuzzyAlgorithm.LEVENSHTEIN
) -> str

def fuzzy_similarity_score(s1: str, s2: str, algorithm: FuzzyAlgorithm) -> float
```

### Usage
```python
from pysearch.search.fuzzy import fuzzy_pattern, FuzzyAlgorithm

# Generate fuzzy regex pattern
fuzzy_regex = fuzzy_pattern("handler", max_distance=2)

# Search with fuzzy matching
results = engine.search(fuzzy_regex, regex=True)
```

---

## Semantic Search (semantic.py & semantic_advanced.py)

### Basic Semantic Search
Lightweight semantic search without external dependencies:
- Token-based similarity
- TF-IDF vectorization
- Cosine similarity scoring

### Advanced Semantic Search
Embedding-based semantic search with:
- Multiple embedding provider support
- Vector database integration
- Multi-modal scoring (text + structure + embeddings)

### Key Functions
```python
# Basic semantic
def semantic_similarity_score(text1: str, text2: str) -> float
def concept_to_patterns(concept: str) -> list[str]

# Advanced semantic
class SemanticEngine:
    async def search(self, query: str, threshold: float = 0.1) -> list[SearchItem]
```

### Usage
```python
# Basic semantic search
from pysearch.search.semantic import semantic_similarity_score

score = semantic_similarity_score("database connection", "db connect")

# Advanced semantic search
results = await engine.search_semantic_advanced(
    "database connection",
    threshold=0.2,
    max_results=100
)
```

---

## Result Scoring (scorer.py)

### Overview
The `scorer.py` module implements result ranking and scoring algorithms.

### Ranking Strategies
- **DEFAULT**: Balanced scoring across all factors
- **RELEVANCE**: Favor exact matches
- **FREQUENCY**: Favor frequent patterns
- **RECENCY**: Favor recently modified files
- **POPULARITY**: Favor commonly accessed files
- **HYBRID**: Combine multiple signals

### Key Functions
```python
def sort_items(
    items: list[SearchItem],
    config: SearchConfig,
    pattern: str,
    strategy: RankingStrategy = RankingStrategy.HYBRID
) -> list[SearchItem]

def deduplicate_overlapping_results(items: list[SearchItem]) -> list[SearchItem]

def cluster_results_by_similarity(
    items: list[SearchItem],
    threshold: float = 0.8
) -> list[list[SearchItem]]
```

### Scoring Factors
- **Match Quality**: Exact vs. partial matches
- **Context Relevance**: Surrounding code context
- **File Importance**: File size, modification time
- **Structural Weight**: AST vs. text matches
- **Frequency**: Match count and distribution

---

## Boolean Logic (boolean.py)

### Overview
The `boolean.py` module provides boolean query evaluation with logical operators.

### Supported Operators
- **AND**: Both conditions must match
- **OR**: Either condition must match
- **NOT**: First condition must match, second must not
- **Parentheses**: Grouping for complex queries

### Key Functions
```python
def parse_boolean_query(query: str) -> BooleanQuery

def evaluate_boolean_query_with_items(
    query: BooleanQuery,
    content: str,
    items: list[SearchItem]
) -> list[SearchItem]
```

### Usage
```python
# Boolean search
results = engine.search(
    "(async AND handler) NOT test",
    use_boolean=True
)
```

---

## Dependencies

### Internal Dependencies
- `pysearch.core`: Types and configuration
- `pysearch.analysis`: Language detection
- `pysearch.utils`: Text processing utilities

### External Dependencies
- `regex`: Enhanced regex support
- `rapidfuzz`: Fast fuzzy matching
- `sklearn`: TF-IDF vectorization (for semantic)
- `numpy`: Numerical operations

---

## Testing

### Unit Tests
Located in `tests/unit/core/`:
- `test_matchers_min.py` - Basic matcher tests
- `test_fuzzy_comprehensive.py` - Fuzzy search tests
- `test_semantic_advanced_*.py` - Semantic search tests
- `test_scorer_advanced.py` - Scoring tests

---

## Common Usage Patterns

### Basic Pattern Matching
```python
from pysearch import PySearch, SearchConfig, Query

config = SearchConfig(paths=["."])
engine = PySearch(config)

# Text search
results = engine.search("def main")

# Regex search
results = engine.search(r"class \w+Test", regex=True)

# AST search with filters
from pysearch.types import ASTFilters
filters = ASTFilters(func_name=".*handler")
query = Query(pattern="def", use_ast=True, ast_filters=filters)
results = engine.run(query)
```

### Fuzzy Search
```python
# Fuzzy search for approximate matches
results = engine.fuzzy_search(
    "handler",
    max_distance=2,
    min_similarity=0.6
)
```

### Semantic Search
```python
# Basic semantic search
results = engine.semantic_search("database connection")

# Advanced semantic search
results = await engine.search_semantic_advanced(
    "web api endpoint",
    threshold=0.2,
    max_results=50
)
```

### Boolean Search
```python
# Boolean query with logical operators
results = engine.search(
    "(async AND handler) NOT test",
    use_boolean=True
)
```

---

## Related Files
- `README.md` - Module overview
- `docs/architecture.md` - Architecture details
- `docs/api/matchers.md` - Matcher API reference
