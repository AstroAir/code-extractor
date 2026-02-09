# Scorer API

The scorer module provides result ranking and scoring functionality to prioritize search results by relevance.

## Main Functions

### score_item

::: pysearch.search.scorer.score_item
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### sort_items

::: pysearch.search.scorer.sort_items
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### deduplicate_overlapping_results

::: pysearch.search.scorer.deduplicate_overlapping_results
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

### cluster_results_by_similarity

::: pysearch.search.scorer.cluster_results_by_similarity
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      docstring_style: google

## Ranking Strategy

### RankingStrategy

::: pysearch.search.scorer.RankingStrategy
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Usage Examples

### Basic Scoring

```python
from pysearch import SearchConfig
from pysearch.search.scorer import score_item, sort_items

config = SearchConfig(
    ast_weight=2.0,
    text_weight=1.0
)

# Score individual item
score = score_item(search_item, config, query_text="def main")
print(f"Item score: {score:.2f}")

# Sort all results by score
sorted_items = sort_items(search_results.items, config, "def main")
```

### Custom Ranking Weights

```python
# Emphasize AST matches over text matches
config = SearchConfig(
    ast_weight=3.0,
    text_weight=1.0,
    rank_strategy=RankingStrategy.DEFAULT
)

# Score considers:
# - Text match frequency and quality
# - Match density and distribution  
# - Code structure (functions, classes, comments)
# - File type relevance
# - Semantic similarity
# - File popularity indicators
# - Position and context bonuses
```

### Deduplication

```python
from pysearch.scorer import deduplicate_overlapping_results

# Remove overlapping results from same file
deduplicated = deduplicate_overlapping_results(
    search_results.items,
    overlap_threshold=0.8
)

print(f"Reduced from {len(search_results.items)} to {len(deduplicated)} results")
```

### Result Clustering

```python
from pysearch.scorer import cluster_results_by_similarity

# Group similar results together
clusters = cluster_results_by_similarity(
    search_results.items,
    similarity_threshold=0.7
)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} similar results")
```

## Scoring Factors

The scoring system considers multiple factors:

### Text Matching
- **Exact matches**: Higher scores for exact pattern matches
- **Case sensitivity**: Bonus for case-sensitive matches
- **Match frequency**: More matches = higher score
- **Match density**: Concentrated matches score higher

### Code Structure
- **Function definitions**: Bonus for matches in function signatures
- **Class definitions**: Bonus for matches in class definitions
- **Import statements**: Bonus for matches in import statements
- **Comments/documentation**: Lower bonus for comment matches

### File Characteristics
- **File type**: Language-specific relevance scoring
- **File size**: Balanced scoring (not too small, not too large)
- **Directory depth**: Penalty for deeply nested files
- **File popularity**: Bonus for frequently accessed files

### Position Context
- **Early position**: Bonus for matches near file beginning
- **Important sections**: Bonus for matches in key code sections
- **Context quality**: Better context around matches

### Semantic Similarity
- **Concept matching**: Bonus for semantically related content
- **Identifier similarity**: Bonus for similar variable/function names
- **Domain relevance**: Context-aware semantic scoring

## Advanced Scoring

### Custom Scoring Functions

```python
def custom_scorer(item, config, query_text="", all_files=None):
    base_score = score_item(item, config, query_text, all_files)
    
    # Add custom scoring logic
    if "test" in str(item.file):
        base_score *= 0.8  # Lower priority for test files
    
    if any("TODO" in line for line in item.lines):
        base_score *= 1.2  # Higher priority for TODO items
    
    return base_score

# Use custom scorer
custom_sorted = sorted(
    search_results.items,
    key=lambda item: custom_scorer(item, config, "def main"),
    reverse=True
)
```

### Ranking Strategies

Different ranking strategies optimize for different use cases:

- **DEFAULT**: Balanced scoring for general use
- **PRECISION**: Emphasizes exact matches and code structure
- **RECALL**: Broader matching with semantic similarity
- **SPEED**: Faster scoring with fewer factors

## Performance Considerations

### Scoring Performance
- Scoring is O(n) where n is the number of results
- Semantic scoring adds computational overhead
- File popularity tracking requires additional I/O

### Memory Usage
- Clustering requires O(n²) memory for similarity matrix
- Deduplication is memory-efficient O(n)
- Large result sets may need streaming processing

## Related

- [Types](types.md) - SearchItem and scoring types
- [Configuration](config.md) - Ranking configuration options
- [Performance](../guide/performance.md) - Performance optimization guide
