# Enhanced Code Indexing Engine Architecture

## Overview

This document outlines the architecture for the enhanced code indexing engine inspired by Continue's implementation, building upon the existing pysearch foundation.

## Current Architecture Analysis

### Existing Strengths
- **Multi-language support**: Already supports 20+ programming languages
- **Sophisticated caching**: JSON-based caching with mtime/size/SHA1 tracking
- **File watching**: Real-time file change detection with debouncing
- **Multi-repository support**: Parallel search across multiple repositories
- **Semantic search**: Lightweight semantic matching without external models
- **Metadata indexing**: Entity-level indexing with SQLite storage
- **GraphRAG integration**: Knowledge graph capabilities

### Gaps Identified (Continue's Advantages)
1. **Content addressing**: Continue uses SHA256 hashes as cache keys for better deduplication
2. **Tag-based indexing**: Branch/directory/artifact tagging for version management
3. **Global cache**: Cross-branch caching to avoid duplicate work
4. **Incremental updates**: More sophisticated diffing with compute/delete/addTag/removeTag operations
5. **Vector database integration**: Better LanceDB/vector database support
6. **Tree-sitter chunking**: Code-aware chunking that respects language structure
7. **Batch processing**: Memory-efficient batch processing for large codebases
8. **Progress tracking**: Detailed progress updates with pause/resume capability

## Enhanced Architecture Design

### 1. Content-Addressed Index System

```python
@dataclass
class ContentAddress:
    """SHA256-based content addressing for efficient caching."""
    path: str
    content_hash: str  # SHA256 of file contents
    size: int
    mtime: float
    
@dataclass
class IndexTag:
    """Tag system for managing index versions."""
    directory: str
    branch: str
    artifact_id: str
    
    def to_string(self) -> str:
        return f"{self.directory}::{self.branch}::{self.artifact_id}"
```

### 2. Multi-Index Architecture

```python
class EnhancedCodebaseIndex(ABC):
    """Base class for all index types."""
    
    @property
    @abstractmethod
    def artifact_id(self) -> str:
        """Unique identifier for this index type."""
        pass
    
    @property
    @abstractmethod
    def relative_expected_time(self) -> float:
        """Relative time cost for this index type."""
        pass
    
    @abstractmethod
    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: str | None = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """Update the index with new/changed/deleted files."""
        pass
```

### 3. Index Types

#### Code Snippets Index
- **Purpose**: Extract top-level code structures (functions, classes, etc.)
- **Technology**: Tree-sitter queries for multiple languages
- **Storage**: SQLite with content and metadata
- **Enhancement**: Support for 20+ languages vs Continue's limited set

#### Full-Text Search Index
- **Purpose**: Fast text-based search across all content
- **Technology**: SQLite FTS5 with trigram tokenization
- **Storage**: SQLite virtual table
- **Enhancement**: Better ranking and filtering capabilities

#### Chunk Index
- **Purpose**: Intelligent code chunking for embeddings
- **Technology**: Language-aware chunking with tree-sitter
- **Storage**: SQLite with chunk metadata
- **Enhancement**: Multi-language chunking strategies

#### Vector Database Index
- **Purpose**: Semantic similarity search using embeddings
- **Technology**: Multiple embedding providers + vector databases
- **Storage**: LanceDB/Qdrant/Chroma with SQLite metadata
- **Enhancement**: Support for multiple vector databases and embedding providers

#### Dependency Graph Index
- **Purpose**: Code dependency and relationship tracking
- **Technology**: AST analysis + import resolution
- **Storage**: SQLite with graph structure
- **Enhancement**: Cross-language dependency tracking

### 4. Incremental Update System

```python
@dataclass
class RefreshIndexResults:
    """Results of index refresh operation."""
    compute: list[PathAndCacheKey]      # New files to index
    delete: list[PathAndCacheKey]       # Files to remove completely
    add_tag: list[PathAndCacheKey]      # Existing content, new tag
    remove_tag: list[PathAndCacheKey]   # Remove tag, keep content
    
class IndexRefreshEngine:
    """Manages incremental index updates."""
    
    async def refresh_index(
        self,
        tag: IndexTag,
        current_files: dict[str, FileStats],
        read_file: Callable[[str], Awaitable[str]],
    ) -> RefreshIndexResults:
        """Compute what needs to be updated in the index."""
        pass
```

### 5. Global Cache System

```python
class GlobalCacheManager:
    """Cross-branch content cache to avoid duplicate work."""
    
    async def get_cached_content(
        self,
        content_hash: str,
        artifact_id: str,
    ) -> Any | None:
        """Get cached content by hash and artifact type."""
        pass
    
    async def store_cached_content(
        self,
        content_hash: str,
        artifact_id: str,
        content: Any,
        tags: list[IndexTag],
    ) -> None:
        """Store content in global cache with associated tags."""
        pass
```

### 6. Enhanced Language Support

```python
class LanguageProcessor:
    """Language-specific processing for enhanced indexing."""
    
    @abstractmethod
    async def chunk_code(
        self,
        content: str,
        max_chunk_size: int,
    ) -> AsyncGenerator[CodeChunk, None]:
        """Language-aware code chunking."""
        pass
    
    @abstractmethod
    def extract_entities(self, content: str) -> list[CodeEntity]:
        """Extract code entities (functions, classes, etc.)."""
        pass
    
    @abstractmethod
    def analyze_dependencies(self, content: str) -> list[Dependency]:
        """Analyze code dependencies and imports."""
        pass
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Implement content-addressed caching system
2. Build tag-based index management
3. Create enhanced base classes and interfaces

### Phase 2: Index Implementations
1. Enhance existing indexes with new architecture
2. Implement vector database integration
3. Build dependency graph index

### Phase 3: Advanced Features
1. Implement distributed indexing
2. Add comprehensive error handling
3. Build performance monitoring

### Phase 4: Testing and Documentation
1. Create comprehensive test suite
2. Write detailed documentation
3. Performance benchmarking

## Benefits Over Continue

1. **Multi-language first**: Native support for 20+ languages vs Continue's limited set
2. **Multiple vector databases**: Support for LanceDB, Qdrant, Chroma vs Continue's LanceDB only
3. **Advanced semantic search**: Both lightweight and embedding-based semantic search
4. **GraphRAG integration**: Built-in knowledge graph capabilities
5. **Dependency analysis**: Cross-language dependency tracking
6. **Better error handling**: Comprehensive error collection and recovery
7. **Performance monitoring**: Built-in metrics and optimization tools
