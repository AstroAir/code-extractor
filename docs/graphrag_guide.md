# GraphRAG Guide for pysearch

## Overview

GraphRAG (Graph Retrieval-Augmented Generation) is a powerful enhancement to pysearch that combines traditional code search with knowledge graph-based analysis and vector similarity search. This guide covers all aspects of GraphRAG functionality, from basic setup to advanced usage patterns.

## Table of Contents

1. [What is GraphRAG?](#what-is-graphrag)
2. [Key Features](#key-features)
3. [Installation and Setup](#installation-and-setup)
4. [Basic Usage](#basic-usage)
5. [Configuration](#configuration)
6. [Qdrant Integration](#qdrant-integration)
7. [Metadata Indexing](#metadata-indexing)
8. [Advanced Queries](#advanced-queries)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## What is GraphRAG?

GraphRAG extends traditional text-based code search by:

- **Building Knowledge Graphs**: Automatically extracting entities (functions, classes, variables) and their relationships from code
- **Semantic Understanding**: Using vector embeddings to understand code semantics beyond keyword matching
- **Graph Traversal**: Finding related code through relationship networks (e.g., "find all functions that call this API")
- **Context-Aware Results**: Providing rich context about code relationships and dependencies

### Traditional Search vs GraphRAG

| Traditional Search | GraphRAG |
|-------------------|----------|
| Keyword/regex matching | Semantic understanding |
| File-based results | Entity-based results |
| Limited context | Rich relationship context |
| Static patterns | Dynamic graph traversal |

## Key Features

### 🔍 **Entity Extraction**
- Automatic detection of functions, classes, methods, variables, imports
- Multi-language support (Python, JavaScript, TypeScript, Java, C#)
- Metadata extraction (signatures, docstrings, complexity metrics)

### 🕸️ **Relationship Mapping**
- Function calls and method invocations
- Class inheritance and composition
- Import dependencies
- Variable usage and data flow

### 🧠 **Semantic Search**
- Vector embeddings for code similarity
- Natural language queries ("find database connection code")
- Semantic clustering of related functionality

### 📊 **Metadata Indexing**
- Comprehensive metadata storage
- Incremental updates
- Performance analytics
- SQLite-based persistence

### ⚡ **Vector Database Integration**
- Qdrant support for scalable vector search
- Configurable embedding models
- Efficient similarity queries

## Installation and Setup

### Basic Installation

```bash
# Install pysearch with GraphRAG dependencies
pip install pysearch[graphrag]

# Or install with all optional dependencies
pip install pysearch[all]
```

### Optional Dependencies

```bash
# For Qdrant vector database support
pip install qdrant-client

# For enhanced vector operations
pip install numpy

# For custom embedding models
pip install sentence-transformers
```

### Qdrant Setup (Optional)

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or using Docker Compose
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

## Basic Usage

### Simple GraphRAG Search

```python
import asyncio
from pysearch import PySearch, SearchConfig, GraphRAGQuery, EntityType

async def basic_graphrag_example():
    # Configure with GraphRAG enabled
    config = SearchConfig(
        paths=["./src"],
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )

    # Initialize search engine
    search = PySearch(config=config, enable_graphrag=True)

    # Build knowledge graph
    await search.build_knowledge_graph()

    # Perform GraphRAG query
    query = GraphRAGQuery(
        pattern="database connection",
        entity_types=[EntityType.FUNCTION, EntityType.CLASS],
        max_hops=2
    )

    results = await search.graphrag_search(query)

    # Process results
    for entity in results.entities:
        print(f"Found: {entity.name} ({entity.entity_type})")
        print(f"File: {entity.file_path}:{entity.start_line}")
        if entity.docstring:
            print(f"Doc: {entity.docstring[:100]}...")

    await search.close_async_components()

# Run the example
asyncio.run(basic_graphrag_example())
```

### Hybrid Search

```python
async def hybrid_search_example():
    config = SearchConfig(
        paths=["./src"],
        enable_graphrag=True,
        enable_enhanced_indexing=True
    )
    
    search = PySearch(config=config, enable_graphrag=True)
    await search.build_knowledge_graph()
    
    # Hybrid search combines multiple approaches
    results = await search.hybrid_search(
        pattern="error handling",
        use_graphrag=True,
        use_enhanced_index=True,
        graphrag_max_hops=2
    )
    
    print(f"Methods used: {results['metadata']['methods_used']}")
    print(f"Traditional matches: {len(results['traditional']['items'])}")
    print(f"GraphRAG entities: {len(results['graphrag']['entities'])}")
    
    await search.close_async_components()
```

## Configuration

### Basic Configuration

```python
from pysearch import SearchConfig

config = SearchConfig(
    # Basic paths
    paths=["./src", "./lib"],
    include=["**/*.py", "**/*.js"],
    exclude=["**/node_modules/**"],

    # Enable GraphRAG features
    enable_graphrag=True,
    enable_enhanced_indexing=True,

    # GraphRAG parameters
    graphrag_max_hops=2,
    graphrag_min_confidence=0.7,
    graphrag_semantic_threshold=0.8,
    graphrag_context_window=5
)
```

### Advanced Configuration

```python
config = SearchConfig(
    paths=["./src"],
    
    # GraphRAG settings
    enable_graphrag=True,
    graphrag_max_hops=3,
    graphrag_min_confidence=0.6,
    graphrag_semantic_threshold=0.75,
    graphrag_context_window=10,
    
    # Enhanced indexing
    enable_enhanced_indexing=True,
    enhanced_indexing_include_semantic=True,
    enhanced_indexing_complexity_analysis=True,
    enhanced_indexing_dependency_tracking=True,
    
    # Qdrant integration
    qdrant_enabled=True,
    qdrant_host="localhost",
    qdrant_port=6333,
    qdrant_collection_name="my_project",
    qdrant_vector_size=384,
    qdrant_distance_metric="Cosine",
    
    # Performance tuning
    max_workers=8,
    cache_ttl=3600,
    timeout=60.0
)
```

## Qdrant Integration

### Setup with Qdrant

```python
from pysearch.qdrant_client import QdrantConfig

# Qdrant configuration
qdrant_config = QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="pysearch_vectors",
    vector_size=384,
    distance_metric="Cosine"
)

# Initialize with Qdrant
search = PySearch(
    config=config,
    qdrant_config=qdrant_config,
    enable_graphrag=True
)
```

### Production Qdrant Setup

```python
qdrant_config = QdrantConfig(
    host="qdrant.example.com",
    port=6333,
    api_key="your-api-key",
    https=True,
    timeout=60.0,
    collection_name="production_vectors",
    vector_size=768,
    distance_metric="Cosine",
    max_retries=5,
    batch_size=200,
    enable_compression=True
)
```

## Metadata Indexing

### Building Metadata Index

```python
# Build with semantic analysis
await search.build_enhanced_index(
    include_semantic=True,
    force_rebuild=False
)
```

### Querying Metadata Index

```python
from pysearch.indexer_metadata import IndexQuery

# Query by entity characteristics
query = IndexQuery(
    entity_types=["function", "class"],
    languages=["python"],
    min_complexity=10.0,
    has_docstring=True,
    limit=20
)

results = await search.enhanced_index_search(query)

print(f"Found {len(results['entities'])} entities")
for entity in results['entities']:
    print(f"- {entity['name']} (complexity: {entity['complexity']})")
```

### File-based Queries

```python
# Query by file characteristics
query = IndexQuery(
    languages=["python"],
    min_size=1000,
    max_size=50000,
    min_lines=50,
    modified_after=1640995200  # Unix timestamp
)

results = await search.enhanced_index_search(query)
```

## Advanced Queries

### Multi-hop Graph Traversal

```python
# Find functions and their call chains
query = GraphRAGQuery(
    pattern="authentication",
    entity_types=[EntityType.FUNCTION],
    relation_types=[RelationType.CALLS, RelationType.USES],
    max_hops=3,
    include_relationships=True
)

results = await search.graphrag_search(query)

# Analyze call chains
for relationship in results.relationships:
    if relationship.relation_type == RelationType.CALLS:
        print(f"Call: {relationship.source_entity_id} -> {relationship.target_entity_id}")
```

### Semantic Similarity Search

```python
# Find semantically similar code
query = GraphRAGQuery(
    pattern="handle user input validation",
    use_vector_search=True,
    semantic_threshold=0.8,
    max_hops=1
)

results = await search.graphrag_search(query)

for entity in results.entities:
    score = results.similarity_scores.get(entity.id, 0.0)
    print(f"{entity.name}: {score:.3f} similarity")
```

### Complex Filtering

```python
# Advanced filtering example
query = GraphRAGQuery(
    pattern="database operations",
    entity_types=[EntityType.FUNCTION, EntityType.METHOD],
    relation_types=[RelationType.CALLS, RelationType.CONTAINS],
    max_hops=2,
    min_confidence=0.8,
    semantic_threshold=0.75,
    context_window=15
)
```

## Performance Optimization

### Large Codebase Configuration

```python
config = SearchConfig(
    paths=["./src"],
    
    # Optimized for large codebases
    graphrag_max_hops=2,  # Limit traversal depth
    graphrag_min_confidence=0.8,  # Higher confidence threshold
    graphrag_context_window=5,  # Smaller context window
    
    # Qdrant optimization
    qdrant_batch_size=500,  # Larger batches
    qdrant_vector_size=256,  # Smaller vectors
    
    # Performance settings
    max_workers=16,
    cache_ttl=7200,  # 2 hours
    timeout=120.0
)
```

### Real-time Search Configuration

```python
config = SearchConfig(
    paths=["./src"],
    
    # Optimized for speed
    graphrag_max_hops=1,  # Single hop only
    graphrag_semantic_threshold=0.9,  # High precision
    graphrag_context_window=3,  # Minimal context
    
    # Fast indexing
    enhanced_indexing_complexity_analysis=False,
    enhanced_indexing_dependency_tracking=False,
    
    # Aggressive caching
    cache_ttl=3600,
    max_workers=8
)
```

### Memory Optimization

```python
# For memory-constrained environments
config = SearchConfig(
    paths=["./src"],
    
    # Reduce memory usage
    max_file_size=5 * 1024 * 1024,  # 5MB limit
    qdrant_vector_size=128,  # Smaller vectors
    qdrant_batch_size=50,  # Smaller batches
    
    # Limit concurrent processing
    max_workers=4,
    
    # Shorter cache retention
    cache_ttl=900  # 15 minutes
)
```

## Troubleshooting

### Common Issues

#### GraphRAG Not Finding Results

```python
# Check if knowledge graph was built
if search._graphrag_engine:
    graph = search._graphrag_engine.graph_builder.knowledge_graph
    print(f"Entities: {len(graph.entities)}")
    print(f"Relationships: {len(graph.relationships)}")
else:
    print("GraphRAG engine not initialized")

# Lower thresholds for broader results
query = GraphRAGQuery(
    pattern="your search",
    semantic_threshold=0.5,  # Lower threshold
    min_confidence=0.3,      # Lower confidence
    max_hops=3               # More hops
)
```

#### Qdrant Connection Issues

```python
# Test Qdrant connection
from pysearch.qdrant_client import QdrantVectorStore

try:
    vector_store = QdrantVectorStore(qdrant_config)
    await vector_store.initialize()
    print("✓ Qdrant connection successful")
    await vector_store.close()
except Exception as e:
    print(f"✗ Qdrant connection failed: {e}")
```

#### Performance Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check configuration
issues = config.validate_advanced_config()
if issues:
    for issue in issues:
        print(f"Config issue: {issue}")
```

### Debug Information

```python
# Get detailed statistics
if search._enhanced_indexer:
    stats = await search._enhanced_indexer.metadata_index.get_stats()
    print(f"Index stats: {stats}")

# Check entity context
if search._graphrag_engine:
    context = await search._graphrag_engine.get_entity_context("entity_id")
    print(f"Entity context: {context}")
```

## Best Practices

1. **Start Simple**: Begin with basic GraphRAG configuration and gradually add features
2. **Build Incrementally**: Use `force_rebuild=False` to avoid rebuilding unchanged graphs
3. **Tune Thresholds**: Adjust semantic and confidence thresholds based on your codebase
4. **Monitor Performance**: Use appropriate batch sizes and worker counts for your hardware
5. **Cache Effectively**: Set appropriate cache TTL values for your usage patterns
6. **Validate Configuration**: Always check for configuration issues before deployment

## Examples

See the `examples/` directory for comprehensive examples:

- `graphrag_examples.py` - Complete usage examples
- `graphrag_configuration.py` - Configuration examples for different scenarios

## API Reference

For detailed API documentation, see the individual module documentation:

- `pysearch.graphrag` - Entity extraction and relationship mapping
- `pysearch.graphrag_engine` - Main GraphRAG engine
- `pysearch.qdrant_client` - Qdrant vector database integration
- `pysearch.indexer_metadata` - Metadata indexing system
- `pysearch.types` - GraphRAG data types and queries
