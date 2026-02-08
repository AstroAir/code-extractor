"""
pysearch: High-performance, context-aware search engine for Python codebases.

This package provides a comprehensive search solution for Python projects, supporting
text/regex/AST/semantic search with both CLI and programmable API interfaces. It's
designed for engineering-grade retrieval in large multi-module projects with
performance optimizations and intelligent caching.

Key Features:
    - **Multiple Search Modes**: Text, regex, AST structural, and semantic search
    - **Code Block Matching**: Functions, classes, decorators, imports, strings/comments
    - **Context-Aware Output**: Configurable context lines around matches
    - **Project-Wide Indexing**: Efficient caching for large codebases with incremental updates
    - **Advanced Filtering**: Include/exclude patterns, AST filters, metadata filters
    - **Performance Optimized**: Parallel processing, directory pruning, smart caching
    - **Rich Output Formats**: Plain text, JSON, highlighted console output
    - **Result Ranking**: Configurable scoring and similarity clustering
    - **Dual Interfaces**: Both CLI and programmatic API access
    - **Error Handling**: Comprehensive error collection and reporting
    - **Extensible**: Plugin-ready architecture for custom matchers and filters

Main Classes:
    PySearch: Main search engine class that orchestrates all operations
    SearchConfig: Configuration object for search parameters and performance tuning
    Query: Search query specification with filters and options
    SearchResult: Search results container with items and performance statistics
    ASTFilters: AST-based filtering for structural code matching
    MetadataFilters: File metadata-based filtering (size, date, language, etc.)

Core Modules:
    api: Main search engine and high-level interfaces
    config: Configuration management and validation
    types: Core data types and enumerations
    matchers: Pattern matching implementations (text, regex, AST, semantic)
    indexer: File indexing and caching system
    scorer: Result ranking and similarity algorithms
    formatter: Output formatting and highlighting
    cli: Command-line interface implementation

Example Usage:
    Basic API usage:
        >>> from pysearch import PySearch, SearchConfig
        >>> config = SearchConfig(paths=["."], include=["**/*.py"])
        >>> engine = PySearch(config)
        >>> results = engine.search(pattern="def main", regex=True)
        >>> for item in results.items:
        ...     print(f"{item.file}: {len(item.lines)} lines")

    Advanced query with filters:
        >>> from pysearch.types import Query, ASTFilters
        >>> filters = ASTFilters(func_name=".*handler", decorator="lru_cache")
        >>> query = Query(pattern="def", use_ast=True, filters=filters)
        >>> results = engine.run(query)

    CLI usage:
        $ pysearch find --pattern "def main" --path . --regex --context 2
        $ pysearch find --pattern "class.*Test" --regex --ast --filter-decorator "pytest.*"

Performance Features:
    - Parallel file processing with configurable worker threads
    - Incremental indexing based on file modification times
    - Optional strict hash checking for exact change detection
    - Directory pruning to skip excluded paths during traversal
    - In-memory caching with TTL for frequently accessed files
    - Smart result deduplication and similarity clustering

Use Cases:
    - Code refactoring and analysis
    - Finding deprecated API usage
    - Security auditing and vulnerability scanning
    - Technical debt identification
    - Dependency analysis and import tracking
    - Code quality assessment
    - Documentation and comment analysis

For detailed documentation, examples, and advanced usage patterns, see:
    - docs/guide/usage.md for getting started
    - docs/guide/configuration.md for configuration options
    - docs/advanced/architecture.md for implementation details
    - examples/ directory for practical examples
    - Project repository: https://github.com/AstroAir/pysearch
"""

from .analysis.language_detection import detect_language, get_supported_languages
from .core.api import PySearch
from .core.config import SearchConfig
from .core.history import SearchHistory
from .core.types import (
    ASTFilters,
    CodeEntity,
    EntityRelationship,
    EntityType,
    FileMetadata,
    GraphRAGQuery,
    GraphRAGResult,
    KnowledgeGraph,
    Language,
    MatchSpan,
    MetadataFilters,
    OutputFormat,
    Query,
    RelationType,
    SearchItem,
    SearchResult,
    SearchStats,
)
from .utils.error_handling import (
    EncodingError,
    FileAccessError,
    ParsingError,
    PermissionError,
    SearchError,
)
from .utils.logging_config import (
    configure_logging,
    disable_logging,
    enable_debug_logging,
    get_logger,
)
from .utils.metadata_filters import create_metadata_filters

# Metadata indexing functionality (optional)
try:
    from .indexing.advanced.engine import IndexingEngine  # noqa: F401
    from .indexing.advanced.integration import (  # noqa: F401
        IndexSearchEngine,
        IndexSearchResult,
        index_search,
        ensure_indexed,
    )

    METADATA_INDEXING_AVAILABLE = True
except ImportError:
    METADATA_INDEXING_AVAILABLE = False

# Storage functionality (optional)
try:
    from .storage import qdrant_client

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    qdrant_client = None  # type: ignore

# Indexing functionality
try:
    from .indexing import indexer

    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False
    indexer = None  # type: ignore

# Package metadata
__version__ = "0.1.0"
__author__ = "Max Qian"
__email__ = "astro_air@126.com"
__license__ = "MIT"
__description__ = "High-performance, context-aware search engine for Python codebases"
__url__ = "https://github.com/AstroAir/pysearch"

# Public API
__all__ = [
    # Main classes
    "PySearch",
    "SearchConfig",
    "SearchHistory",
    # Data types
    "OutputFormat",
    "SearchItem",
    "SearchStats",
    "SearchResult",
    "Query",
    "ASTFilters",
    "MatchSpan",
    "Language",
    "FileMetadata",
    "MetadataFilters",
    "GraphRAGQuery",
    "GraphRAGResult",
    "KnowledgeGraph",
    "CodeEntity",
    "EntityRelationship",
    "EntityType",
    "RelationType",
    # Utility functions
    "detect_language",
    "get_supported_languages",
    "create_metadata_filters",
    # Logging and configuration
    "configure_logging",
    "get_logger",
    "enable_debug_logging",
    "disable_logging",
    # Exception classes
    "SearchError",
    "FileAccessError",
    "PermissionError",
    "EncodingError",
    "ParsingError",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
    # Metadata indexing availability
    "METADATA_INDEXING_AVAILABLE",
    "QDRANT_AVAILABLE",
    "INDEXER_AVAILABLE",
    "qdrant_client",
    "indexer",
]

# Add metadata indexing functionality to __all__ if available
if METADATA_INDEXING_AVAILABLE:
    __all__.extend(
        [
            "IndexSearchEngine",
            "IndexSearchResult",
            "IndexingEngine",
            "index_search",
            "ensure_indexed",
        ]
    )
