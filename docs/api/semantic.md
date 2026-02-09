# Semantic Search API

The semantic search module provides lightweight semantic matching capabilities without requiring external models.

## Main Functions

### semantic_similarity_score

::: pysearch.search.semantic.semantic_similarity_score
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### concept_to_patterns

::: pysearch.search.semantic.concept_to_patterns
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### expand_semantic_query

::: pysearch.search.semantic.expand_semantic_query
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Advanced Semantic Search

### SemanticSearchEngine

::: pysearch.search.semantic_advanced.SemanticSearchEngine
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
      group_by_category: true
      show_bases: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Usage Examples

### Basic Semantic Search

```python
from pysearch.semantic import semantic_similarity_score

# Compare code content with query
content = """
def connect_database():
    conn = sqlite3.connect('app.db')
    return conn
"""

query = "database connection"
score = semantic_similarity_score(content, query)
print(f"Semantic similarity: {score:.2f}")
```

### Concept-Based Search

```python
from pysearch.semantic import get_concept_patterns

# Get patterns for database-related concepts
db_patterns = get_concept_patterns("database")
print("Database patterns:", db_patterns)

# Search for web-related concepts
web_patterns = get_concept_patterns("web")
print("Web patterns:", web_patterns)
```

### Feature Extraction

```python
from pysearch.semantic import extract_semantic_features

code = """
async def fetch_user_data(user_id):
    async with aiohttp.ClientSession() as session:
        response = await session.get(f'/api/users/{user_id}')
        return await response.json()
"""

features = extract_semantic_features(code)
print("Semantic features:", features)
```

### Advanced Semantic Engine

```python
from pysearch.semantic_advanced import SemanticSearchEngine

# Initialize semantic engine
semantic_engine = SemanticSearchEngine()

# Add documents to index
semantic_engine.add_document("doc1", "Database connection handling")
semantic_engine.add_document("doc2", "Web API client implementation")
semantic_engine.add_document("doc3", "Async task processing")

# Search semantically
results = semantic_engine.search("database operations", top_k=5)
for doc_id, score in results:
    print(f"{doc_id}: {score:.3f}")
```

## Concept Categories

The semantic search system recognizes several concept categories:

### Database Operations
- Connection management
- Query execution
- Transaction handling
- ORM operations

```python
# Matches: db, database, conn, connection, cursor, execute, query
# Also: sql, mysql, postgres, sqlite, mongodb, redis
# And: session, transaction, commit, rollback
```

### Web Development
- HTTP operations
- API endpoints
- Web frameworks
- Authentication

```python
# Matches: http, https, request, response, get, post, put, delete
# Also: flask, django, fastapi, tornado, bottle
# And: route, endpoint, api, rest, json, xml
```

### Testing
- Test functions
- Assertions
- Mocking
- Test frameworks

```python
# Matches: test, assert, mock, patch, fixture, setUp, tearDown
# Also: pytest, unittest, nose, doctest
# And: should, expect, verify, check, validate
```

### Asynchronous Programming
- Async/await patterns
- Coroutines
- Event loops
- Task management

```python
# Matches: async, await, asyncio, coroutine, future, task
# Also: async def, await expressions
# And: gather, create_task, run, get_event_loop
```

## Customization

### Custom Concept Patterns

```python
from pysearch.semantic import CONCEPT_PATTERNS

# Add custom concept patterns
CONCEPT_PATTERNS["machine_learning"] = [
    r"\b(ml|machine_learning|sklearn|tensorflow|pytorch)\b",
    r"\b(model|train|predict|fit|score)\b",
    r"\b(neural|network|deep|learning)\b"
]

# Use in semantic search
score = semantic_similarity_score(code, "machine learning")
```

### Semantic Weights

```python
from pysearch import SearchConfig

# Adjust semantic scoring weight
config = SearchConfig(
    semantic_weight=1.5,  # Increase semantic importance
    text_weight=1.0,
    ast_weight=2.0
)
```

## Performance Considerations

### Lightweight Design
- No external dependencies (no transformers, no embeddings)
- Pattern-based matching using regex
- Fast concept recognition
- Minimal memory footprint

### Scalability
- O(1) concept pattern matching
- Linear scaling with content size
- Efficient caching of semantic features
- Suitable for large codebases

### Accuracy Trade-offs
- Simpler than transformer-based models
- Good for code-specific concepts
- May miss complex semantic relationships
- Optimized for developer workflows

## Integration with Search

### Query Processing

```python
from pysearch.types import Query

# Semantic query
query = Query(
    pattern="database connection",
    use_semantic=True,
    context=5
)

# Combined semantic + text search
query = Query(
    pattern="connect",
    use_regex=True,
    use_semantic=True  # Boost semantically related matches
)
```

### Result Scoring

Semantic similarity contributes to overall result scoring:

1. **Text matching**: Direct pattern matches
2. **AST matching**: Structural code matches  
3. **Semantic matching**: Conceptual relevance
4. **Combined score**: Weighted sum of all factors

## Related

- [Matchers](matchers.md) - Core matching functionality
- [Scorer](scorer.md) - Result scoring and ranking
- [Types](types.md) - Query and result types
- [Examples](../../examples/README.md) - Semantic search examples
