# Enhanced Code Indexing Engine Guide

## Overview

The Enhanced Code Indexing Engine is a sophisticated code analysis and search system inspired by Continue's indexing architecture but with significant enhancements and broader capabilities. It provides comprehensive code understanding through multiple indexing strategies, advanced language support, and intelligent caching.

## Key Features

### 🚀 **Enhanced Architecture**
- **Content-Addressed Caching**: SHA256-based content addressing for efficient incremental updates
- **Tag-Based Index Management**: Branch/directory/artifact tagging for version management
- **Global Cache**: Cross-branch caching to avoid duplicate indexing work
- **Multi-Index Architecture**: Coordinated multiple index types for comprehensive coverage

### 🌍 **Advanced Language Support**
- **20+ Programming Languages**: Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, and more
- **Tree-Sitter Integration**: Advanced AST-based parsing and analysis
- **Language-Specific Chunking**: Intelligent code chunking that respects language boundaries
- **Semantic Analysis**: Code structure understanding and entity extraction

### 🔍 **Multiple Index Types**
- **Code Snippets Index**: Function/class/variable extraction with metadata
- **Full-Text Search Index**: SQLite FTS5 for fast text-based search
- **Chunk Index**: Intelligent code chunking for embeddings and analysis
- **Vector Index**: Semantic similarity search using embeddings
- **Dependency Index**: Code relationship and dependency tracking

### ⚡ **Performance & Scalability**
- **Distributed Indexing**: Multi-process parallel indexing for large codebases
- **Incremental Updates**: Smart diffing to minimize reindexing work
- **Memory Optimization**: Efficient memory usage with streaming processing
- **Performance Monitoring**: Real-time metrics and optimization suggestions

### 🛡️ **Reliability & Recovery**
- **Enhanced Error Handling**: Comprehensive error categorization and recovery
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Automatic Recovery**: Self-healing capabilities for common failures
- **Health Monitoring**: System health assessment and recommendations

## Quick Start

### Installation

```bash
# Install enhanced indexing dependencies
pip install tree-sitter tree-sitter-python tree-sitter-javascript
pip install lancedb  # For vector indexing
pip install openai   # For embeddings (optional)
```

### Basic Usage

```python
from pysearch.enhanced_indexing_engine import EnhancedIndexingEngine
from pysearch.config import SearchConfig

# Create configuration
config = SearchConfig(
    paths=["./src", "./lib"],
    cache_dir="./cache",
    enable_enhanced_indexing=True,
)

# Initialize and run indexing
engine = EnhancedIndexingEngine(config)
await engine.initialize()

# Index codebase with progress tracking
async for progress in engine.refresh_index():
    print(f"Progress: {progress.progress:.1%} - {progress.description}")
```

### Advanced Configuration

```python
from pysearch.enhanced_indexing_engine import IndexCoordinator
from pysearch.indexes import EnhancedCodeSnippetsIndex, EnhancedVectorIndex
from pysearch.enhanced_vector_db import EmbeddingConfig

# Custom embedding configuration
embedding_config = EmbeddingConfig(
    provider="openai",
    model_name="text-embedding-ada-002",
    batch_size=50,
    api_key="your-api-key"
)

# Create coordinator with custom indexes
coordinator = IndexCoordinator(config)
coordinator.add_index(EnhancedCodeSnippetsIndex(config))
coordinator.add_index(EnhancedVectorIndex(config))

# Index with custom settings
tag = IndexTag("./src", "main", "enhanced_search")
async for progress in coordinator.refresh_all_indexes(tag, files, read_file):
    print(f"Indexing: {progress.description}")
```

## Architecture Deep Dive

### Content Addressing System

The enhanced indexing engine uses SHA256-based content addressing similar to Git:

```python
@dataclass
class ContentAddress:
    path: str
    content_hash: str  # SHA256 of file contents
    size: int
    mtime: float
    language: Language
```

**Benefits:**
- Exact change detection (no false positives from timestamp changes)
- Cross-branch content sharing through global cache
- Efficient incremental updates
- Deduplication of identical content

### Tag-Based Index Management

Indexes are organized using a three-part tag system:

```python
@dataclass
class IndexTag:
    directory: str    # Repository/directory path
    branch: str       # Git branch name
    artifact_id: str  # Index type identifier
```

**Benefits:**
- Multiple index versions for different branches
- Efficient branch switching without reindexing
- Granular index management and cleanup
- Support for monorepo scenarios

### Multi-Index Architecture

The system coordinates multiple specialized indexes:

1. **Code Snippets Index** (`enhanced_code_snippets`)
   - Extracts functions, classes, variables using tree-sitter
   - Stores entity metadata (signatures, docstrings, complexity)
   - Enables precise code structure search

2. **Full-Text Search Index** (`enhanced_full_text`)
   - SQLite FTS5 with trigram tokenization
   - Fast text-based search across all content
   - Language and file type filtering

3. **Chunk Index** (`enhanced_chunks`)
   - Intelligent code-aware chunking
   - Respects function/class boundaries
   - Optimized for embedding generation

4. **Vector Index** (`enhanced_vectors`)
   - Semantic similarity search using embeddings
   - Multiple vector database backends
   - Advanced retrieval with re-ranking

## Language Support

### Supported Languages

| Language | Tree-Sitter | Entity Extraction | Chunking | Status |
|----------|-------------|-------------------|----------|---------|
| Python | ✅ | ✅ | ✅ | Full Support |
| JavaScript | ✅ | ✅ | ✅ | Full Support |
| TypeScript | ✅ | ✅ | ✅ | Full Support |
| Java | ✅ | ✅ | ✅ | Full Support |
| C/C++ | ✅ | ✅ | ✅ | Full Support |
| Go | ✅ | ✅ | ✅ | Full Support |
| Rust | ✅ | ✅ | ✅ | Full Support |
| PHP | ⚠️ | ⚠️ | ✅ | Partial Support |
| Ruby | ⚠️ | ⚠️ | ✅ | Partial Support |
| C# | ⚠️ | ⚠️ | ✅ | Partial Support |

### Adding New Languages

```python
from pysearch.enhanced_language_support import LanguageProcessor, language_registry

class CustomLanguageProcessor(LanguageProcessor):
    async def chunk_code(self, content: str, max_chunk_size: int):
        # Implement language-specific chunking
        pass
    
    def extract_entities(self, content: str):
        # Implement entity extraction
        pass

# Register custom processor
language_registry.register_processor(Language.CUSTOM, CustomLanguageProcessor())
```

## Performance Optimization

### Chunking Strategies

Choose the optimal chunking strategy based on your use case:

```python
from pysearch.advanced_chunking import ChunkingConfig, ChunkingStrategy

# For speed (large codebases)
fast_config = ChunkingConfig(
    strategy=ChunkingStrategy.STRUCTURAL,
    max_chunk_size=1500,
    respect_boundaries=True,
)

# For quality (smaller codebases)
quality_config = ChunkingConfig(
    strategy=ChunkingStrategy.HYBRID,
    max_chunk_size=1000,
    overlap_size=100,
    quality_threshold=0.8,
)
```

### Distributed Indexing

For very large codebases, use distributed indexing:

```python
from pysearch.distributed_indexing import DistributedIndexingEngine

# Create distributed engine
engine = DistributedIndexingEngine(
    config,
    num_workers=8,  # Adjust based on CPU cores
)

# Index with parallel processing
async for progress in engine.index_codebase(directories):
    print(f"Distributed indexing: {progress.description}")
```

### Memory Management

Configure memory usage for your environment:

```python
# For memory-constrained environments
config = SearchConfig(
    embedding_batch_size=25,    # Smaller batches
    chunk_size=800,             # Smaller chunks
    file_size_limit=1024*1024,  # 1MB file limit
)

# For high-memory environments
config = SearchConfig(
    embedding_batch_size=200,   # Larger batches
    chunk_size=2000,            # Larger chunks
    enable_parallel_processing=True,
)
```

## Vector Database Integration

### Supported Providers

- **LanceDB** (Recommended): High-performance vector database with SQL interface
- **Qdrant**: Scalable vector database with advanced filtering
- **Chroma**: Simple vector database for development and testing

### Configuration

```python
from pysearch.enhanced_vector_db import EmbeddingConfig, VectorIndexManager

# OpenAI embeddings with LanceDB
embedding_config = EmbeddingConfig(
    provider="openai",
    model_name="text-embedding-ada-002",
    api_key="your-api-key",
    batch_size=100,
)

vector_manager = VectorIndexManager(
    db_path=Path("./vector_db"),
    embedding_config=embedding_config,
    provider="lancedb"
)
```

## Error Handling and Recovery

### Automatic Recovery

The system includes automatic recovery for common failures:

```python
from pysearch.utils.advanced_error_handling import EnhancedErrorHandler

error_handler = EnhancedErrorHandler(config)

# Errors are automatically categorized and recovery is attempted
try:
    await risky_operation()
except Exception as e:
    recovery_success = await error_handler.handle_error(
        context="operation_name",
        exception=e,
        attempt_recovery=True
    )
```

### Error Categories

- **File Access**: Permission errors, missing files
- **Network**: API failures, connection timeouts
- **Memory**: Out of memory, allocation failures
- **Parsing**: Syntax errors, malformed content
- **Dependencies**: Missing packages, import errors

## Performance Monitoring

### Real-Time Metrics

```python
from pysearch.performance_monitoring import PerformanceMonitor

monitor = PerformanceMonitor(config, cache_dir)
await monitor.start_monitoring()

# Get performance report
report = await monitor.get_performance_report()
print(f"System health: {report['health_score']:.2f}")
```

### Profiling Operations

```python
from pysearch.performance_monitoring import PerformanceProfiler

profiler = PerformanceProfiler(metrics_collector)

async with profiler.profile_operation("custom_indexing") as profile_id:
    # Your indexing operation here
    await index_large_codebase()
    
    # Update progress
    await profiler.update_profile_stats(profile_id, files_processed=100)
```

## Comparison with Continue

### Improvements Over Continue

| Feature | Continue | Enhanced Engine | Improvement |
|---------|----------|-----------------|-------------|
| Languages | 8 languages | 20+ languages | 2.5x more languages |
| Vector DBs | LanceDB only | LanceDB, Qdrant, Chroma | Multiple providers |
| Chunking | Basic tree-sitter | Advanced hybrid chunking | Better quality chunks |
| Error Handling | Basic warnings | Comprehensive recovery | Production-grade reliability |
| Monitoring | Limited | Real-time metrics | Full observability |
| Distributed | No | Multi-process | Scalability for large codebases |
| Caching | Basic | Global cross-branch | Better cache efficiency |

### Compatibility

The enhanced engine maintains compatibility with existing pysearch APIs while adding new capabilities:

```python
# Existing pysearch usage still works
from pysearch.search import search_files
results = search_files("function_name", ["./src"])

# Enhanced features available through new APIs
from pysearch.enhanced_indexing_engine import EnhancedIndexingEngine
engine = EnhancedIndexingEngine(config)
await engine.refresh_index()
```

## Best Practices

### 1. Index Organization
- Use separate tags for different branches
- Regular cleanup of unused indexes
- Monitor index sizes and performance

### 2. Performance Tuning
- Start with default settings
- Monitor performance metrics
- Apply suggested optimizations
- Scale workers based on CPU cores

### 3. Error Management
- Enable automatic recovery
- Monitor error trends
- Review error logs regularly
- Implement custom recovery strategies for domain-specific errors

### 4. Memory Management
- Adjust batch sizes based on available memory
- Use streaming for large files
- Monitor memory usage trends
- Enable garbage collection optimizations

## Troubleshooting

### Common Issues

**High Memory Usage**
```python
# Reduce batch sizes
config.embedding_batch_size = 25
config.chunk_size = 800
```

**Slow Indexing**
```python
# Enable parallel processing
config.enable_parallel_processing = True
engine = DistributedIndexingEngine(config, num_workers=8)
```

**Network Errors**
```python
# Configure retry and timeout settings
embedding_config.timeout = 30.0
recovery_manager.circuit_breakers["embedding_api"].failure_threshold = 3
```

### Debug Mode

```python
import logging
logging.getLogger("pysearch").setLevel(logging.DEBUG)

# Enable detailed progress tracking
async for progress in engine.refresh_index():
    if progress.debug_info:
        print(f"Debug: {progress.debug_info}")
```

## Migration Guide

### From Basic pysearch

1. **Update Configuration**
```python
# Old configuration
config = SearchConfig(paths=["./src"])

# Enhanced configuration
config = SearchConfig(
    paths=["./src"],
    enable_enhanced_indexing=True,
    embedding_provider="openai",
    vector_db_provider="lancedb",
)
```

2. **Update Indexing Code**
```python
# Old indexing
from pysearch.indexer import Indexer
indexer = Indexer(config)
indexer.index_files()

# Enhanced indexing
from pysearch.enhanced_indexing_engine import EnhancedIndexingEngine
engine = EnhancedIndexingEngine(config)
await engine.initialize()
async for progress in engine.refresh_index():
    print(f"Progress: {progress.progress:.1%}")
```

3. **Update Search Code**
```python
# Enhanced search with multiple index types
from pysearch.indexes import EnhancedCodeSnippetsIndex, EnhancedVectorIndex

# Search code snippets
snippets_index = EnhancedCodeSnippetsIndex(config)
tag = IndexTag("./src", "main", "enhanced_code_snippets")
entities = await snippets_index.retrieve("function_name", tag)

# Semantic search
vector_index = EnhancedVectorIndex(config)
tag = IndexTag("./src", "main", "enhanced_vectors")
similar_code = await vector_index.retrieve("database connection", tag)
```

## API Reference

### Core Classes

#### EnhancedIndexingEngine
Main entry point for enhanced indexing operations.

```python
class EnhancedIndexingEngine:
    async def initialize() -> None
    async def refresh_index(directories, branch, repo_name) -> AsyncGenerator[IndexingProgressUpdate, None]
    async def refresh_file(file_path, directory, branch, repo_name) -> None
    def pause() -> None
    def resume() -> None
    def cancel() -> None
```

#### IndexCoordinator
Coordinates multiple index types.

```python
class IndexCoordinator:
    def add_index(index: EnhancedCodebaseIndex) -> None
    def remove_index(artifact_id: str) -> bool
    def get_index(artifact_id: str) -> Optional[EnhancedCodebaseIndex]
    async def refresh_all_indexes(...) -> AsyncGenerator[IndexingProgressUpdate, None]
```

#### ContentAddress
Content-addressed file representation.

```python
@dataclass
class ContentAddress:
    path: str
    content_hash: str  # SHA256
    size: int
    mtime: float
    language: Language
```

#### IndexTag
Tag system for index management.

```python
@dataclass
class IndexTag:
    directory: str
    branch: str
    artifact_id: str
    
    def to_string() -> str
    @classmethod
    def from_string(tag_string: str) -> IndexTag
```

### Index Types

#### EnhancedCodeSnippetsIndex
```python
async def retrieve(query, tag, limit=50, **kwargs) -> List[Dict[str, Any]]
async def search_entities(query, tag, entity_types, languages, min_quality, limit) -> List[Dict[str, Any]]
async def get_entities_by_file(file_path, tag) -> List[Dict[str, Any]]
```

#### EnhancedVectorIndex
```python
async def retrieve(query, tag, limit=50, **kwargs) -> List[Dict[str, Any]]
async def get_similar_chunks(chunk_content, tag, limit=10) -> List[Dict[str, Any]]
async def rerank_results(results, query, boost_factors) -> List[Dict[str, Any]]
```

#### EnhancedFullTextIndex
```python
async def retrieve(query, tag, limit=50, **kwargs) -> List[Dict[str, Any]]
async def search_in_file(file_path, query, tag, context_lines=3) -> List[Dict[str, Any]]
```

## Examples

### Example 1: Basic Enhanced Indexing

```python
import asyncio
from pysearch.enhanced_indexing_engine import EnhancedIndexingEngine
from pysearch.config import SearchConfig

async def main():
    # Configure enhanced indexing
    config = SearchConfig(
        paths=["./src"],
        cache_dir="./cache",
        enable_enhanced_indexing=True,
    )
    
    # Initialize engine
    engine = EnhancedIndexingEngine(config)
    await engine.initialize()
    
    # Index codebase
    print("Starting enhanced indexing...")
    async for progress in engine.refresh_index():
        print(f"[{progress.progress:.1%}] {progress.description}")
        
        if progress.warnings:
            for warning in progress.warnings:
                print(f"Warning: {warning}")
    
    print("Indexing complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Custom Index Configuration

```python
from pysearch.enhanced_indexing_engine import IndexCoordinator
from pysearch.indexes import EnhancedCodeSnippetsIndex, EnhancedVectorIndex
from pysearch.enhanced_vector_db import EmbeddingConfig

async def setup_custom_indexing():
    config = SearchConfig(paths=["./src"])
    
    # Configure embeddings
    embedding_config = EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-ada-002",
        batch_size=50,
    )
    
    # Create coordinator
    coordinator = IndexCoordinator(config)
    
    # Add only specific indexes
    coordinator.add_index(EnhancedCodeSnippetsIndex(config))
    
    # Conditionally add vector index
    if embedding_config.api_key:
        coordinator.add_index(EnhancedVectorIndex(config))
    
    return coordinator
```

### Example 3: Performance Monitoring

```python
from pysearch.performance_monitoring import PerformanceMonitor

async def monitor_indexing_performance():
    config = SearchConfig(paths=["./large_codebase"])
    monitor = PerformanceMonitor(config, Path("./cache"))
    
    # Start monitoring
    await monitor.start_monitoring()
    
    try:
        # Perform indexing
        engine = EnhancedIndexingEngine(config)
        await engine.initialize()
        
        async for progress in engine.refresh_index():
            if progress.progress % 0.1 < 0.01:  # Every 10%
                report = await monitor.get_performance_report()
                print(f"Health Score: {report['health_score']:.2f}")
    
    finally:
        await monitor.stop_monitoring()
```

## Next Steps

1. **Install Dependencies**: Install required packages for your use case
2. **Configure Settings**: Set up configuration for your codebase
3. **Run Initial Index**: Perform initial indexing of your codebase
4. **Monitor Performance**: Use performance monitoring to optimize settings
5. **Integrate with Tools**: Integrate with your development workflow

For more detailed examples and advanced usage patterns, see the `examples/` directory.
